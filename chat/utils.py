import os
import datetime
import uuid
import re
import json
import requests
import pandas as pd

import chromadb
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(
    model=gemini_model,
    temperature=0.7,
)

# Maintain Google Gemini model (Ollama reverted)
sql_llm = llm

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
chroma_data_path = os.path.join(os.path.expanduser("~"), ".sqlchat", "chroma_db")
os.makedirs(chroma_data_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_data_path)

sessions_collection  = chroma_client.get_or_create_collection(name="chat_sessions")
messages_collection  = chroma_client.get_or_create_collection(name="chat_messages")
charts_collection    = chroma_client.get_or_create_collection(name="chat_charts")
feedback_collection  = chroma_client.get_or_create_collection(name="chat_feedback")

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
dataset_intro_cache = {
    "text": None,
    "analysis": None,
    "suggestions": None,
    "updated_at": None,
}
chart_context_store = {}


import oracledb
import urllib.parse
from langchain_community.utilities import SQLDatabase

# Oracle 11.2 requires Thick mode in python-oracledb
try:
    oracledb.init_oracle_client()
except Exception:
    try:
        # Update path to point to your extracted 64-bit Oracle Instant Client
        instant_client_path = r"C:\instantclient_19\instantclient_21_22"
        oracledb.init_oracle_client(lib_dir=instant_client_path)
    except Exception as e:
        print(f"Warning: Failed to load Oracle Client in Thick Mode: {e}")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_database(user, password, database) -> SQLDatabase:
    if not user or not password:
        raise ValueError("Missing database credentials! Please configure your .env file.")
    
    # URL-encode the password to handle any special characters safely
    safe_password = urllib.parse.quote_plus(password)
    # Connection string for your local Oracle instance
    db_uri = f"oracle+oracledb://{user}:{safe_password}@localhost:1521/?service_name={database}"
    return SQLDatabase.from_uri(db_uri)

# ---------------------------------------------------------------------------
# Currency Conversion
# ---------------------------------------------------------------------------
exchange_rate_cache = {
    "rates_text": "",
    "updated_at": None
}

def get_exchange_rates_context():
    now = datetime.datetime.now()
    if exchange_rate_cache["rates_text"] and exchange_rate_cache["updated_at"]:
        if (now - exchange_rate_cache["updated_at"]).total_seconds() < 3600 * 12: # 12 hours
            return exchange_rate_cache["rates_text"]
            
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            rates = data.get("rates", {})
            major_currencies = ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "MXN", "BRL", "UGX", "KES", "RWF"]
            lines = [f"1 USD = {rates[c]} {c}" for c in major_currencies if c in rates]
            rates_text = "Live Exchange Rates (Base 1 USD):\n" + "\n".join(lines)
            
            exchange_rate_cache["rates_text"] = rates_text
            exchange_rate_cache["updated_at"] = now
            return rates_text
    except Exception:
        pass
        
    return "Live exchange rates unavailable at the moment."


# ---------------------------------------------------------------------------
# LLM Chains
# ---------------------------------------------------------------------------
def get_sql_chain(db):
    template = """
        You are an expert Oracle Database engineer. Generate precise SQL queries that answer user questions, and handle cases where data might not exist.

        CRITICAL INSTRUCTIONS:
        - If the user's request does NOT require querying the database (e.g., small talk, greetings, or general questions),
          output exactly: NO_SQL
        - Otherwise output ONLY the SQL query without any explanations, prefixes, or suffixes
        - Do not format the query with new lines - keep it as a single line
        - Do NOT end the query with a semicolon (;). Oracle driver via SQLAlchemy will throw ORA-00911 for trailing semicolons.
        - Use Oracle Database syntax exclusively (compatible with Oracle 11.2). Remember we are using Oracle DB, NOT Microsoft SQL Server. Do not use SQL Server specific syntax, functions, or operators.
        - Always stepback to transform the query in a generic understanding, then respond effectively to the question.
        - Respond based on the database schema, very important for specific questions!
        - Carry forward any explicit filters or scope from the conversation history unless the user overrides them
        - If the user asks for trends, variance, comparisons (MoM/YoY/QoQ), or drill-downs, include the needed time buckets,
          comparison periods, and grouping dimensions directly in the SQL

        HANDLING SPECIFIC SEARCHES AND NULL RESULTS:
        - When asked about specific entities (like car models, product names, etc.), search through relevant text columns using LIKE or = operators
        - If the user mentions a specific name (e.g., "Corolla"), search for exact matches first, then consider partial matches
        - Always structure queries to return a count or result even if the data doesn't exist - the query should execute successfully regardless
        - Use COALESCE or NVL to handle potential null values when needed

        INTELLIGENT SCHEMA MAPPING:
        - Analyze the schema carefully to identify columns that might contain the requested information
        - Look for columns with names like: name, model, title, description, type, category, brand, make, etc.
        - When searching for specific terms, consider multiple relevant columns
        - Use OR conditions to search across multiple relevant columns when appropriate
        - Oracle tables and columns are stored in UPPERCASE. Always write table and column names in UPPERCASE without double quotes.

        ORACLE DATABASE SPECIFIC SYNTAX (Oracle 11.2):
        - IMPORTANT: Do not use SQL Server functions like GETDATE() or DATEADD() or syntax like SELECT TOP.
        - DO NOT use LIMIT or TOP. To restrict row counts, use ROWNUM.
          Example for top rows: SELECT * FROM (SELECT * FROM table ORDER BY col DESC) WHERE ROWNUM <= 10
        - Use SYSDATE for current date/time
        - Use || for string concatenation (e.g., first_name || ' ' || last_name)
        - Use LIKE for partial text matching: WHERE model_name LIKE '%COROLLA%'
        - Use UPPER() or LOWER() for case-insensitive searches: WHERE UPPER(model_name) = 'COROLLA'
        - CLOB DATATYPES HANDLING:
          Many text columns in the database (like COUNTRY, REGION, PRODUCTNAME, FIRSTNAME, etc.) are of type CLOB.
          In Oracle, you CANNOT perform GROUP BY, ORDER BY, DISTINCT, or JOINS directly on CLOB columns.
          You MUST convert CLOB columns using TO_CHAR(column_name) or CAST(column_name AS VARCHAR2(4000)) in these clauses.
          Example: SELECT TO_CHAR(country), COUNT(*) FROM table GROUP BY TO_CHAR(country)
          Example: ORDER BY TO_CHAR(productname)
          Always apply TO_CHAR() or CAST() to any CLOB columns when grouping, ordering, filtering, or joining.

        QUERY STRUCTURE FOR SPECIFIC SEARCHES:
        - For "how many X" questions: SELECT COUNT(*) FROM table WHERE conditions
        - For existence checks: SELECT CASE WHEN EXISTS (subquery) THEN 1 ELSE 0 END FROM dual
        - Always make the query executable regardless of whether data exists

        VISUAL / CHART REQUESTS:
        - If the user asks for a chart, graph, or visual, you MUST still output a SQL query to fetch the underlying data.
        - Do NOT return NO_SQL for chart requests — return the SQL that retrieves the data to be charted.
        - Use the conversation history to understand what data the user is referring to (e.g. "piechart for that" = re-query the last topic).

        [SUPPLEMENTARY EXCHANGE RATES]
        The following exchange rates are available for currency conversion calculations:
        {currency_context}

        Conversation History: {chat_history}

        Database Schema, very important for sql writing: {schema}

        User Question: {question}

        Generate the Oracle SQL query (or NO_SQL):
        """

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return RunnablePassthrough.assign(schema=get_schema, currency_context=lambda _: get_exchange_rates_context()) | prompt | sql_llm | StrOutputParser()


def build_no_sql_response(user_query, chat_history):
    template = """
        You are a helpful assistant for a business intelligence chatbot.
        The user question does not require SQL. Provide a concise, helpful response.
        - Use conversation history to retain context and scope unless the user overrides them
        - Always return TWO sections in this order:
          1) Response: a direct, effective answer to the user question
          2) Suggestive analysis: 3-6 SHORT question-style follow-ups (one per line), under 20 words each

        [SUPPLEMENTARY EXCHANGE RATES]
        The following exchange rates are available for currency conversion if the user asks for it. Do NOT treat this as the main dataset.
        {currency_context}

        Question:{question}
        Conversation History:{chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query, 
        "chat_history": chat_history,
        "currency_context": get_exchange_rates_context()
    })


def is_data_overview_question(user_query):
    q = (user_query or "").lower().strip()
    direct_patterns = [
        r"\bwhat is (this|the)?\s*data (all about|about)\b",
        r"\bdescribe (this|the)?\s*data(set)?\b",
        r"\bwhat does (this|the) data (show|contain|represent)\b",
        r"\boverview of (this|the)?\s*data(set)?\b",
        r"\bsummar(y|ize) (this|the)?\s*data(set)?\b",
        r"\bgive me (an )?(overview|summary) of (this|the)?\s*data(set)?\b",
    ]
    if any(re.search(p, q) for p in direct_patterns):
        return True
    has_data_word = bool(re.search(r"\b(data|dataset|database|schema|tables?)\b", q))
    has_overview_intent = bool(
        re.search(r"\b(about|overview|summary|summarize|describe|high[- ]level|big picture)\b", q)
    )
    return has_data_word and has_overview_intent


def is_sql_query_question(user_query):
    q = (user_query or "").lower().strip()
    if is_data_overview_question(q):
        return False
        
    sql_indicators = [
        r"\b(revenue|sales|orders?|products?|customers?|countries|territories|cost|price|quantity)\b",
        r"\b(trend|monthly|yearly|weekly|quarterly|daily)\b",
        r"\b(how\s+many|total|average|avg|sum|max|min|count|percent|growth|difference)\b",
        r"\b(select|show\s+me|list|generate|plot|chart|graph|table)\b",
        r"\b(from\s+20\d{2}|since\s+20\d{2}|between\s+20\d{2}|in\s+20\d{2})\b"
    ]
    if any(re.search(pattern, q) for pattern in sql_indicators):
        return True
    return False


def build_data_overview_response(user_query, db, chat_history):
    template = """
        You are a data analyst assistant.
        The user is asking for what the dataset is about.
        Write exactly two concise paragraphs (4-6 sentences) that explains:
        - what business/domain the data likely covers (important, with potential KPIs)
        - key entities/tables and how they relate at a high level
        - important metrics/dimensions the user can analyze
        - one brief caveat if schema context is limited
        Use the schema provided and infer likely domain from table/column names.
        Never ask the user which tables to choose.
        If schema context is limited, still provide the best possible high-level overview.
        Do not output bullet points.
        Keep it plain and readable.

        User Question: {question}
        Conversation History: {chat_history}
        Database Schema: {schema}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "schema": db.get_table_info(),
    })


def has_table_intent(question):
    q = (question or "").lower()
    patterns = [
        r"\b(show|display|give me|render|present|output|format|return)\b.{0,40}\b(table|tabular|grid|spreadsheet)\b",
        r"\b(as|in|using)\b.{0,20}\b(a |an )?(table|tabular format|grid)\b",
        r"\b(table|tabular)\b.{0,30}\b(form|format|view|layout)\b",
        r"\bput (it|that|the (data|results?|output)) in(to)? a table\b",
        r"\b(list|show).{0,30}\bin a table\b",
        r"\bgenerate (a |an )?table\b",
        r"\bmake (it|that|this) (a )?table\b",
        r"\btabulate\b",
    ]
    return any(re.search(p, q) for p in patterns)


def build_sql_response(user_query, db, chat_history, query, sql_result):
    forced_table = has_table_intent(user_query)

    if forced_table:
        table_override = """
        !!MANDATORY INSTRUCTION Ã¢â‚¬â€ OVERRIDE ALL OTHER FORMATTING RULES!!
        The user has explicitly requested a TABLE. You MUST:
        1. Present ALL the SQL result data as a proper markdown table at the very beginning of your Response (up to 25 rows max).
           Use this exact syntax:
           | Column1 | Column2 | ... |
           | ------- | ------- | ... |
           | value   | value   | ... |
        2. If there are more than 25 rows, show the first 25 and note "(showing first 25 of N rows)".
        3. Do NOT summarise or paraphrase the rows instead of showing the table.
        4. After the table, add 2-3 concise insight sentences.
        Failure to produce a markdown table when the user asked for one is UNACCEPTABLE.
        """
    else:
        table_override = """
        TABLE RENDERING RULE:
        - Only render a markdown table if the result has 20 or fewer rows of data.
        - If the result has more than 20 rows (e.g. time-series trend data across many groups), do NOT render a huge table.
          Instead: summarize the key findings in 3-5 bullet points (e.g. top performer, lowest, biggest growth, anomalies).
        - If the user explicitly asks to "show", "list", "display", "give me", or "what are" the results,
          render a table only if 20 or fewer rows Ã¢â‚¬â€ otherwise still summarize.
        - For single-value or single-row results, no table is needed Ã¢â‚¬â€ just state the value clearly.
        - Use proper markdown table syntax with a header row and separator row (| col | col | and | --- | --- |).
        """

    template = """
        You are a data analyst assistant for a business intelligence chatbot.
        Based on the table schema below, question SQL query and SQL response, write a clear, actionable response.

        {table_instruction}

        ADDITIONAL RULES:
        - Respond based on the data returned by the database
        - Use conversation history to retain context, filters, and scope unless the user overrides them
        - Go beyond raw output: provide insights, comparisons, and recommendations when supported by the data
        - Identify trends, anomalies, and variance where possible; explain likely drivers if the data suggests them
        - Provide contextual comparisons (e.g., MoM/YoY/QoQ) when the data includes time buckets
        - Suggest relevant follow-up analysis and drill-down options
        - If the data is insufficient for a requested insight, say so briefly and suggest what to query next
        - Provide outbound triggers: if the data indicates unusual spikes/drops or breaches of typical patterns, explicitly flag them
        - Keep answers concise and structured
        - SYSTEM CAPABILITY: The system will automatically render a chart if the data supports it. Do NOT apologize for being a text AI. Talk about the chart naturally.
        - Always return TWO sections in this order:
          1) Response: (table if applicable, then narrative insight)
          2) Suggestive analysis: 3-6 SHORT question-style follow-ups (one per line), phrased as user questions.
             These must be actionable drill-downs, comparisons, or next analyses. Keep each under 14 words.

        [SUPPLEMENTARY EXCHANGE RATES]
        The following exchange rates are available for currency conversion:
        {currency_context}

        {schema}

        Question:{question}
        SQL Query:{query}
        Conversation History:{chat_history}
        SQL Response:{response}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "schema": db.get_table_info(),
        "query": query,
        "response": sql_result,
        "table_instruction": table_override,
        "currency_context": get_exchange_rates_context(),
    })


def extract_clean_sql(text):
    if not text:
        return ""
    
    text = text.strip()
    
    # 1. Handle markdown code blocks
    md_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if md_match:
        text = md_match.group(1).strip()
        
    # 2. Split lines and remove comment blocks/fluff
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        l_strip = line.strip()
        if not l_strip:
            continue
        if l_strip.startswith("```"):
            continue
        # Skip SQL inline comment lines
        if l_strip.startswith("--") or l_strip.startswith("#") or l_strip.startswith("//"):
            continue
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines).strip()
    
    # 3. Find the first occurrence of SQL keywords
    match = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|MERGE|EXEC|CREATE|ALTER|DROP)\b[\s\S]*", text, re.IGNORECASE)
    if match:
        sql_part = match.group(0).strip()
        semi_idx = sql_part.find(";")
        if semi_idx != -1:
            sql_part = sql_part[:semi_idx + 1]
        else:
            # If no semicolon is found, truncate at double newlines or explanation patterns
            lines = sql_part.split("\n")
            sql_lines = []
            for line in lines:
                l_strip = line.strip()
                if re.match(r"^(this\s+query|note:|here\s+is|explanation|the\s+query|we\s+can|this\s+statement|to\s+calculate)\b", l_strip, re.IGNORECASE):
                    break
                sql_lines.append(line)
            sql_part = "\n".join(sql_lines).strip()
        return sql_part.strip()
        
    return text


def normalize_response_text(text):
    if not text:
        return text
    cleaned = text.strip()
    label_pattern = r"\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+\s*[\)\.\-:]\s*)?response\s*(?:\*\*|__)?\s*[:\-]?\s*"
    cleaned = re.sub(r"^" + label_pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?im)^[ \t]*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+\s*[\)\.\-:]\s*)?response\s*(?:\*\*|__)?\s*[:\-]?\s*",
        "",
        cleaned,
        count=1,
    )
    return cleaned


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def get_session_messages_sorted(session_id):
    results = messages_collection.get(where={"session_id": session_id})
    messages = []
    if results["ids"]:
        for i, msg_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] or {}
            messages.append({
                "id": msg_id,
                "role": metadata.get("role"),
                "content": results["documents"][i],
                "timestamp": metadata.get("timestamp", ""),
                "metadata": metadata,
            })
    messages.sort(key=lambda x: x["timestamp"])
    return messages


def build_langchain_history(messages):
    chat_history = []
    for m in messages:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "bot":
            chat_history.append(AIMessage(content=m["content"]))
    return chat_history


def get_response_with_query(user_query, db, chat_history):
    if is_data_overview_question(user_query):
        response_text = build_data_overview_response(user_query, db, chat_history)
        return normalize_response_text(response_text), None, None

    sql_chain = get_sql_chain(db)
    query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    }).strip()

    if query == "NO_SQL":
        response_text = build_no_sql_response(user_query, chat_history)
        return normalize_response_text(response_text), None, None

    sql_result = db.run(query)
    response_text = build_sql_response(user_query, db, chat_history, query, sql_result)
    return normalize_response_text(response_text), query, sql_result


# ---------------------------------------------------------------------------
# Dataset intro cache
# ---------------------------------------------------------------------------
def get_default_start_suggestions():
    return [
        "What are the top KPIs in this dataset?",
        "Show monthly trends for the main metrics.",
        "Which categories contribute most to total value?",
        "Where are unusual spikes or drops happening?",
        "Compare current period vs previous period.",
        "Break down performance by region or department.",
    ]


def get_dataset_suggestions(db):
    try:
        schema = db.get_table_info()
        template = """
        You are a business intelligence assistant. Based on the database schema below, generate 5 relevant, concrete, and actionable questions a user might want to ask this dataset.
        
        Database Schema:
        {schema}
        
        RULES:
        - The questions must focus on the main business metrics of the dataset (such as revenue/sales, order quantity, order count, returns/refunds, and profit).
        - Each question must be directly relevant to the actual tables and columns in the schema (e.g. referencing specific columns, tables, products, regions, or metrics found in the schema).
        - Do not generate generic questions like "What are the top KPIs?" or "Compare current period vs previous period". Make them specific to this dataset (e.g., "What are the top selling product categories by total sales?" or "Show the monthly return rates for products").
        - Keep each question short, concise, and under 15 words.
        - Do not include numbering, prefixes, or bullet points. Output exactly one question per line.
        
        List of 5 questions:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        raw_suggestions = chain.invoke({"schema": schema})
        suggestions = [s.strip().strip("-*•").strip() for s in raw_suggestions.split("\n") if s.strip()]
        valid_suggestions = [s for s in suggestions if s and len(s.split()) < 20][:6]
        if len(valid_suggestions) >= 3:
            return valid_suggestions
    except Exception as e:
        print(f"Error generating dynamic suggestions: {e}")
    
    return get_default_start_suggestions()


def get_dataset_intro_payload():
    default_analysis = (
        "You can run trend analysis, comparisons (MoM/YoY/QoQ), drill-downs by category/time/region, "
        "distribution checks, outlier spotting, quick KPI summaries, generate graphs, and share conversation links."
    )
    try:
        now = datetime.datetime.now()
        cached_text = dataset_intro_cache.get("text")
        cached_analysis = dataset_intro_cache.get("analysis")
        cached_suggestions = dataset_intro_cache.get("suggestions")
        cached_at = dataset_intro_cache.get("updated_at")
        if (
            cached_text
            and cached_analysis
            and cached_suggestions
            and cached_at
            and (now - cached_at).total_seconds() < 3600
        ):
            return {
                "text": cached_text,
                "analysis": cached_analysis,
                "suggestions": cached_suggestions,
            }

        db = init_database(
            os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_NAME")
        )
        raw = build_data_overview_response("What is this data all about?", db, chat_history=[])
        cleaned = normalize_response_text(raw or "")
        normalized = re.sub(r"\s+", " ", cleaned).strip()

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        intro = " ".join([s for s in sentences if s][:2]).strip() or normalized
        if not intro:
            intro = "This chatbot helps you query and analyze your data with natural language."

        suggestions = get_dataset_suggestions(db)
        dataset_intro_cache["text"] = intro
        dataset_intro_cache["analysis"] = default_analysis
        dataset_intro_cache["suggestions"] = suggestions
        dataset_intro_cache["updated_at"] = now
        return {"text": intro, "analysis": default_analysis, "suggestions": suggestions}
    except Exception:
        return {
            "text": "This chatbot helps you query and analyze your data with natural language.",
            "analysis": default_analysis,
            "suggestions": get_default_start_suggestions(),
        }


def warm_dataset_intro_cache():
    try:
        payload = get_dataset_intro_payload()
        dataset_intro_cache["text"] = payload.get("text")
        dataset_intro_cache["analysis"] = payload.get("analysis")
        dataset_intro_cache["suggestions"] = payload.get("suggestions")
        dataset_intro_cache["updated_at"] = datetime.datetime.now()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def fetch_dataframe(db, query):
    try:
        engine = getattr(db, "_engine", None) or getattr(db, "engine", None)
        if engine is None:
            return None
        return pd.read_sql_query(query, engine)
    except Exception:
        return None


def try_parse_dates(df):
    if df is None or df.empty:
        return df
    for col in df.columns:
        if df[col].dtypes == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.6:
                df[col] = parsed
    return df


def is_probable_id_column(series, col_name):
    name = (col_name or "").lower()
    # Explicit ID markers in column name
    if any(token in name for token in [" id", "_id", "id_", "code", "key", "uuid", "guid"]):
        return True
    # Never treat measure-sounding columns as IDs
    _MEASURE_TOKENS = [
        "revenue", "sales", "sale", "amount", "total", "count", "profit",
        "qty", "quantity", "price", "cost", "value", "rate", "ratio",
        "margin", "income", "expense", "loss", "gain", "score", "weight",
        "budget", "forecast", "target", "actual", "spend", "volume",
        "units", "number", "num", "sum", "avg", "average", "mean",
    ]
    if any(token in name for token in _MEASURE_TOKENS):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    # Need enough rows for the heuristic to be meaningful
    if len(non_null) < 30:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    is_int_like = pd.api.types.is_integer_dtype(non_null) or (
        pd.api.types.is_float_dtype(non_null) and (non_null % 1 == 0).all()
    )
    # Only flag as ID-like if integers are high-range (e.g. DB sequential IDs)
    val_range = float(non_null.max() - non_null.min()) if len(non_null) > 0 else 0
    return bool(is_int_like and unique_ratio >= 0.95 and val_range > 1000)



def is_probable_time_dimension_column(series, col_name):
    name = (col_name or "").lower()
    if not any(token in name for token in ["year", "month", "quarter", "week", "day", "date", "time", "period"]):
        return False
    if series is None:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        values = []
        for v in non_null:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv.is_integer():
                values.append(int(fv))
        values = sorted(set(values))
        if len(values) <= 20:
            # Calendar year range
            if any(1900 <= v <= 2100 for v in values):
                return True
            if "year" in name and all(1000 <= v <= 3000 for v in values):
                return True
            # Month numbers (1-12)
            if "month" in name and all(1 <= v <= 12 for v in values):
                return True
            # Quarter numbers (1-4)
            if "quarter" in name and all(1 <= v <= 4 for v in values):
                return True
            # Week numbers (1-53)
            if "week" in name and all(1 <= v <= 53 for v in values):
                return True
            # Day numbers (1-31)
            if "day" in name and all(1 <= v <= 31 for v in values):
                return True
        return False
    if pd.api.types.is_object_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return True
    return False


def choose_category_column(df, numeric_cols, time_cols):
    candidates = [c for c in df.columns if c not in numeric_cols and c not in time_cols]
    best_col = None
    best_score = None
    for c in candidates:
        non_null_ratio = df[c].notna().mean()
        nunique = df[c].nunique(dropna=True)
        if nunique < 2:
            continue
        if nunique > 50:
            continue
        score = (abs(nunique - 10), -non_null_ratio)
        if best_score is None or score < best_score:
            best_col = c
            best_score = score
    return best_col


def choose_measure_columns(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    filtered = [
        c
        for c in numeric_cols
        if not is_probable_id_column(df[c], c) and not is_probable_time_dimension_column(df[c], c)
    ]
    if filtered:
        return filtered
    return [c for c in numeric_cols if not is_probable_id_column(df[c], c)]


def prettify_label(label):
    if not label:
        return ""
    text = str(label).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[_\s]+", " ", text)
    text = re.sub(r"^(total|sum|avg|average|mean|count)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bid\b", "ID", text, flags=re.IGNORECASE)
    return text.title()


def build_chart_title(question, chart_type, x=None, y=None, metrics=None, category=None):
    if chart_type == "clustered_bar" and metrics and category:
        pretty_metrics = [prettify_label(m) for m in metrics[:3]]
        return f"{', '.join(pretty_metrics)} by {prettify_label(category)}"
    if chart_type in ["bar", "pie", "donut"] and y and category:
        return f"{prettify_label(y)} by {prettify_label(category)}"
    if chart_type == "scatter" and x and y:
        return f"{prettify_label(y)} vs {prettify_label(x)}"
    if chart_type == "histogram" and y:
        return f"Distribution of {prettify_label(y)}"
    if chart_type == "line":
        if x and metrics and len(metrics) > 1:
            pretty_metrics = [prettify_label(m) for m in metrics[:3]]
            return f"{', '.join(pretty_metrics)} by {prettify_label(x)}"
        if x and y:
            return f"{prettify_label(y)} by {prettify_label(x)}"
        if y:
            return f"Trend of {prettify_label(y)}"
    if y and category:
        return f"{prettify_label(y)} by {prettify_label(category)}"
    if x and y:
        return f"{prettify_label(y)} vs {prettify_label(x)}"
    if metrics and len(metrics) > 0:
        pretty_metrics = [prettify_label(m) for m in metrics[:3]]
        return f"Metric Overview: {', '.join(pretty_metrics)}"
    return "Data Overview"


# ---------------------------------------------------------------------------
# Chart context store
# ---------------------------------------------------------------------------
def register_chart_context(df):
    dataset_id = str(uuid.uuid4())
    chart_context_store[dataset_id] = try_parse_dates(df.copy()) if df is not None else None
    return dataset_id


def get_chart_context(dataset_id):
    if not dataset_id:
        return None
    df = chart_context_store.get(dataset_id)
    if df is None:
        return None
    return df.copy()


def clear_chart_context(dataset_id):
    if dataset_id in chart_context_store:
        chart_context_store.pop(dataset_id, None)


def _ensure_chart_title(title, question, chart_type, x=None, y=None, metrics=None, category=None):
    title = (title or "").strip()
    if title:
        return title
    return build_chart_title(question, chart_type, x=x, y=y, metrics=metrics, category=category)


def _normalize_column_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in [None, ""]]
    return [str(value)]


def _infer_line_series(df, user_query, requested=None):
    requested = [c for c in _normalize_column_list(requested) if c in df.columns]
    measure_cols = choose_measure_columns(df)
    q = (user_query or "").lower()
    comparison_intent = bool(
        re.search(r"\b(compare|comparison|vs|versus|between|both|together|trend|over time|and)\b", q)
    )
    if len(requested) >= 2:
        return [c for c in requested if not is_probable_time_dimension_column(df[c], c)]
    if requested:
        requested = [c for c in requested if not is_probable_time_dimension_column(df[c], c)]
        if len(requested) >= 2:
            return requested
    candidates = [c for c in measure_cols if not is_probable_time_dimension_column(df[c], c)]
    if comparison_intent and len(candidates) >= 2:
        return candidates[:2]
    if len(candidates) >= 1:
        return candidates[:3]
    return requested


def _infer_clustered_bar_series(df, user_query, requested=None):
    requested = [c for c in _normalize_column_list(requested) if c in df.columns]
    measure_cols = choose_measure_columns(df)
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    category_col = choose_category_column(df, measure_cols, time_cols)
    q = (user_query or "").lower()
    compare_intent = bool(re.search(r"\b(compare|comparison|vs|versus|between|both|together|and)\b", q))
    category_intent = bool(
        re.search(r"\b(by|per)\s+[a-z0-9_ ]+\b", q)
        or re.search(r"\b(category|country|countries|region|regions|product|products|segment|segments)\b", q)
    )
    if len(requested) >= 2:
        requested = [c for c in requested if not is_probable_time_dimension_column(df[c], c)]
        if len(requested) >= 2:
            return requested[:3]
    candidates = [c for c in measure_cols if not is_probable_time_dimension_column(df[c], c)]
    if compare_intent and category_intent and category_col and len(candidates) >= 2:
        return candidates[:3]
    if compare_intent and len(candidates) >= 2 and category_col:
        return candidates[:3]
    return requested


def _should_force_clustered_bar(user_query, df):
    q = (user_query or "").lower()
    if not re.search(r"\b(compare|comparison|vs|versus|between|both|together)\b", q):
        return False
    if not re.search(r"\b(by|per)\b", q) and not re.search(r"\b(category|country|region|product|segment|department)\b", q):
        return False
    measure_cols = choose_measure_columns(df)
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    category_col = choose_category_column(df, measure_cols, time_cols)
    return bool(category_col and len([c for c in measure_cols if not is_probable_time_dimension_column(df[c], c)]) >= 2)


# ---------------------------------------------------------------------------
# ECharts colour palettes
# ---------------------------------------------------------------------------
BAR_PALETTE = [
    "#2e7d32", "#4f83cc", "#26a69a", "#8e6bbd", "#f4a261",
    "#78909c", "#66bb6a", "#5c6bc0", "#26c6da", "#b39ddb",
    "#ffb74d", "#90a4ae",
]
CLUSTER_PALETTE = [
    "#2e7d32", "#4f83cc", "#26a69a", "#8e6bbd", "#f4a261", "#78909c",
]

# Shared ECharts theme defaults injected into every option
_ECHARTS_BASE = {
    "backgroundColor": "transparent",
    "textStyle": {"color": "#c5c8d3", "fontFamily": "Plus Jakarta Sans, sans-serif"},
    "tooltip": {
        "trigger": "axis",
        "backgroundColor": "rgba(17,24,39,0.95)",
        "borderColor": "rgba(255,255,255,0.14)",
        "textStyle": {"color": "#f9fafb"},
        "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
    },
    "legend": {
        "textStyle": {"color": "#ffffff"},
        "inactiveColor": "#4b5563",
        "top": "4%",
    },
    "grid": {"left": "3%", "right": "4%", "bottom": "12%", "top": "15%", "containLabel": True},
    "color": BAR_PALETTE,
}


def _echarts_payload(option, chart_type, chart_title):
    """Wrap an ECharts option dict into the chart payload envelope."""
    merged = dict(_ECHARTS_BASE)
    merged.update(option)
    # preserve base colour palette unless overridden
    if "color" not in option:
        merged["color"] = BAR_PALETTE
    return {
        "type": "echarts",
        "chart_type": chart_type,
        "title": chart_title,
        "option": merged,
    }


def render_chart_from_spec(
    df, question, chart_type,
    x_column=None, y_column=None, y_columns=None,
    category_column=None, title=None, max_categories=12,
):
    if df is None or df.empty:
        return None
    try:
        df = try_parse_dates(df.copy())
        measure_cols = choose_measure_columns(df)
        time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        x_column = str(x_column).strip() if x_column and str(x_column).strip() in df.columns else None
        y_column = str(y_column).strip() if y_column and str(y_column).strip() in df.columns else None
        category_column = (
            str(category_column).strip() if category_column and str(category_column).strip() in df.columns else None
        )
        y_columns = [c for c in _normalize_column_list(y_columns) if c in df.columns]

        # ── LINE & AREA ───────────────────────────────────────────────────────────
        if chart_type in ["line", "area"]:
            if not x_column:
                x_column = time_cols[0] if time_cols else (category_column or (df.columns[0] if len(df.columns) else None))
            y_columns = _infer_line_series(df, question, requested=y_columns or ([y_column] if y_column else []))
            if not x_column or not y_columns:
                return None

            # Auto-detect category column if not provided
            if not category_column and len(y_columns) == 1:
                candidate_cats = [
                    c for c in df.columns
                    if c != x_column
                    and c not in y_columns
                    and not is_probable_id_column(df[c], c)
                    and not is_probable_time_dimension_column(df[c], c)
                    and df[c].dtype == object
                    and 2 <= df[c].nunique(dropna=True) <= 12
                ]
                if candidate_cats:
                    category_column = candidate_cats[0]

            # Multi-line pivot: one line per category value
            if category_column and category_column in df.columns and len(y_columns) == 1:
                pivot_candidates = df[category_column].dropna().unique()
                if 2 <= len(pivot_candidates) <= 12:
                    pivot_df = df[[x_column, category_column, y_columns[0]]].dropna(subset=[x_column, category_column]).copy()
                    pivot_df[x_column] = pivot_df[x_column].astype(str)
                    pivot_df = pivot_df.groupby([x_column, category_column], as_index=False)[y_columns[0]].sum(numeric_only=True)
                    pivot_wide = pivot_df.pivot(index=x_column, columns=category_column, values=y_columns[0]).reset_index()
                    try:
                        pivot_wide[x_column] = pd.to_numeric(pivot_wide[x_column])
                        pivot_wide = pivot_wide.sort_values(x_column)
                        pivot_wide[x_column] = pivot_wide[x_column].astype(str)
                    except Exception:
                        pivot_wide = pivot_wide.sort_values(x_column)
                    group_cols = [c for c in pivot_wide.columns if c != x_column]
                    if len(pivot_wide) >= 2 and group_cols:
                        chart_title = _ensure_chart_title(title, question, chart_type, x=x_column, y=y_columns[0], category=category_column)
                        x_data = pivot_wide[x_column].tolist()
                        series = []
                        for i, gc in enumerate(group_cols):
                            s_data = {
                                "name": prettify_label(gc),
                                "type": "line",
                                "smooth": True,
                                "symbol": "circle",
                                "symbolSize": 6,
                                "data": [round(float(v), 4) if v == v else None for v in pivot_wide[gc].tolist()],
                            }
                            if chart_type == "area":
                                s_data["areaStyle"] = {"opacity": 0.25}
                            series.append(s_data)
                        option = {
                            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30, "color": "#c5c8d3"}},
                            "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                            "legend": {"data": [s["name"] for s in series]},
                            "series": series,
                        }
                        return _echarts_payload(option, chart_type, chart_title)

            # Fallback: simple single/multi-series line (no category grouping)
            plot_df = df[[x_column] + y_columns].dropna(subset=[x_column]).copy()
            if plot_df.empty:
                return None
            if not pd.api.types.is_datetime64_any_dtype(plot_df[x_column]):
                plot_df[x_column] = plot_df[x_column].astype(str)
                agg_df = plot_df.groupby(x_column, as_index=False)[y_columns].sum(numeric_only=True).sort_values(x_column)
            else:
                agg_df = plot_df.groupby(x_column, as_index=False)[y_columns].sum(numeric_only=True).sort_values(x_column)
            if len(agg_df) < 2:
                return None
            chart_title = _ensure_chart_title(title, question, chart_type, x=x_column, y=y_columns[0], metrics=y_columns)
            x_data = agg_df[x_column].astype(str).tolist()
            series = []
            for yc in y_columns:
                s_data = {
                    "name": prettify_label(yc),
                    "type": "line",
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 6,
                    "data": [round(float(v), 4) if v == v else None for v in agg_df[yc].tolist()],
                }
                if chart_type == "area":
                    s_data["areaStyle"] = {"opacity": 0.25}
                series.append(s_data)
            option = {
                "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30, "color": "#c5c8d3"}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"data": [s["name"] for s in series]} if len(series) > 1 else {"show": False},
                "series": series,
            }
            return _echarts_payload(option, "line", chart_title)

        # Ã¢â€â‚¬Ã¢â€â‚¬ BAR Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if chart_type == "bar":
            if not category_column:
                candidates = [c for c in df.columns if c not in measure_cols and c not in time_cols]
                category_column = choose_category_column(df, measure_cols, time_cols) or (candidates[0] if candidates else None)
            if not y_column:
                y_column = measure_cols[0] if measure_cols else None
            if not category_column or not y_column:
                return None
            agg_df = (
                df[[category_column, y_column]]
                .dropna(subset=[category_column, y_column])
                .groupby(category_column, as_index=False)[y_column]
                .sum(numeric_only=True)
                .sort_values(y_column, ascending=False)
            )
            if agg_df.empty:
                return None
            if len(agg_df) > max_categories:
                agg_df = agg_df.head(max_categories)
            chart_title = _ensure_chart_title(title, question, "bar", x=None, y=y_column, category=category_column)
            # Horizontal bar: ECharts uses yAxis as category and xAxis as value
            cat_data = agg_df[category_column].astype(str).tolist()[::-1]  # reverse for top-down ordering
            val_data = [round(float(v), 4) if v == v else 0 for v in agg_df[y_column].fillna(0).tolist()][::-1]
            color_list = [
                "#2e7d32", "#4f83cc", "#26a69a", "#8e6bbd", "#f4a261",
                "#78909c", "#66bb6a", "#5c6bc0", "#26c6da", "#b39ddb",
                "#ffb74d", "#90a4ae"
            ]
            bar_series = []
            for idx, value in enumerate(val_data):
                bar_series.append({
                    "value": value,
                    "itemStyle": {
                        "color": color_list[idx % len(color_list)],
                        "borderRadius": [0, 4, 4, 0],
                    },
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}},
                })
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "yAxis": {"type": "category", "data": cat_data, "axisLabel": {"color": "#c5c8d3"}},
                "grid": {"left": "3%", "right": "4%", "bottom": "4%", "top": "8%", "containLabel": True},
                "legend": {"show": False},
                "series": [{
                    "name": prettify_label(y_column),
                    "type": "bar",
                    "data": bar_series,
                }],
            }
            return _echarts_payload(option, "bar", chart_title)

        # ── CLUSTERED BAR & STACKED BAR ──────────────────────────────────────────
        if chart_type in ["clustered_bar", "stacked_bar"]:
            if not category_column:
                candidates = [c for c in df.columns if c not in measure_cols and c not in time_cols]
                category_column = choose_category_column(df, measure_cols, time_cols) or (candidates[0] if candidates else None)
            if not y_columns:
                y_columns = [y_column] if y_column else measure_cols[:3]
            if not category_column or len(y_columns) < 2:
                return None
            plot_df = df[[category_column] + y_columns].dropna(subset=[category_column]).copy()
            if plot_df.empty:
                return None
            agg_df = plot_df.groupby(category_column, as_index=False)[y_columns].sum(numeric_only=True).sort_values(category_column)
            if len(agg_df) > max_categories:
                agg_df = agg_df.head(max_categories)
            if agg_df.empty:
                return None
            chart_title = _ensure_chart_title(title, question, chart_type, y=y_columns[0], metrics=y_columns, category=category_column)
            cat_data = agg_df[category_column].astype(str).tolist()
            series = []
            for i, yc in enumerate(y_columns):
                s_data = {
                    "name": prettify_label(yc),
                    "type": "bar",
                    "data": [round(float(v), 4) if v == v else 0 for v in agg_df[yc].fillna(0).tolist()],
                    "itemStyle": {"borderRadius": [4, 4, 0, 0] if chart_type != "stacked_bar" else [0, 0, 0, 0]},
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}}
                }
                if chart_type == "stacked_bar":
                    s_data["stack"] = "total"
                series.append(s_data)
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "category", "data": cat_data, "axisLabel": {"rotate": 30, "color": "#c5c8d3"}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"data": [s["name"] for s in series]},
                "series": series,
            }
            return _echarts_payload(option, chart_type, chart_title)

        # Ã¢â€â‚¬Ã¢â€â‚¬ PIE / DONUT Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if chart_type in ["pie", "donut"]:
            if not category_column:
                category_column = choose_category_column(df, measure_cols, time_cols)
            if not y_column:
                y_column = measure_cols[0] if measure_cols else None
            if not category_column or not y_column:
                return None
            agg_df = (
                df[[category_column, y_column]]
                .dropna(subset=[category_column, y_column])
                .groupby(category_column, as_index=False)[y_column]
                .sum(numeric_only=True)
                .sort_values(y_column, ascending=False)
            )
            if agg_df.empty:
                return None
            if len(agg_df) > max_categories:
                agg_df = agg_df.head(max_categories)
            total_value = float(agg_df[y_column].sum()) if not agg_df.empty else 0.0
            if total_value <= 0:
                return None
            chart_title = _ensure_chart_title(title, question, chart_type, y=y_column, category=category_column)
            pie_data = [
                {"name": str(row[category_column]), "value": round(float(row[y_column]), 4)}
                for _, row in agg_df.iterrows() if row[y_column] == row[y_column]
            ]
            radius = ["45%", "72%"] if chart_type == "donut" else ["0%", "72%"]
            option = {
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)", "backgroundColor": "rgba(17,24,39,0.95)", "borderColor": "rgba(255,255,255,0.14)", "textStyle": {"color": "#f9fafb"}},
                "legend": {"orient": "vertical", "right": "2%", "top": "center", "textStyle": {"color": "#ffffff"}},
                "grid": None,
                "series": [{
                    "name": prettify_label(y_column),
                    "type": "pie",
                    "radius": radius,
                    "center": ["40%", "55%"],
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"},
                        "label": {"show": True, "color": "#ffffff", "fontWeight": "bold", "formatter": "{b}: {c} ({d}%)"},
                    },
                    "label": {
                        "show": True,
                        "color": "#ffffff",
                        "position": "outside",
                        "formatter": "{b}: {d}%",
                    },
                    "labelLine": {"show": True, "length": 10, "length2": 8, "lineStyle": {"color": "rgba(255,255,255,0.45)"}},
                }],
            }
            return _echarts_payload(option, chart_type, chart_title)

        # Ã¢â€â‚¬Ã¢â€â‚¬ SCATTER Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if chart_type == "scatter":
            if not x_column:
                x_column = measure_cols[0] if measure_cols else None
            if not y_column:
                y_column = measure_cols[1] if len(measure_cols) > 1 else None
            if not x_column or not y_column:
                return None
            plot_df = df[[x_column, y_column]].dropna().copy()
            if plot_df.empty:
                return None
            if len(plot_df) > 2000:
                plot_df = plot_df.sample(2000, random_state=42)
            chart_title = _ensure_chart_title(title, question, "scatter", x=x_column, y=y_column)
            scatter_data = [[round(float(r[x_column]), 4), round(float(r[y_column]), 4)] for _, r in plot_df.iterrows()]
            option = {
                "tooltip": {"trigger": "item", "formatter": f"{prettify_label(x_column)}: {{b}}<br/>{prettify_label(y_column)}: {{c}}"},

                "xAxis": {"type": "value", "name": prettify_label(x_column), "nameLocation": "middle", "nameGap": 30, "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "yAxis": {"type": "value", "name": prettify_label(y_column), "nameLocation": "middle", "nameGap": 40, "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"show": False},
                "series": [{
                    "name": f"{prettify_label(y_column)} vs {prettify_label(x_column)}",
                    "type": "scatter",
                    "data": scatter_data,
                    "symbolSize": 6,
                    "itemStyle": {"opacity": 0.75},
                }],
            }
            return _echarts_payload(option, "scatter", chart_title)

        # Ã¢â€â‚¬Ã¢â€â‚¬ HISTOGRAM Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if chart_type == "histogram":
            if not y_column:
                y_column = measure_cols[0] if measure_cols else None
            if not y_column:
                return None
            plot_df = df[[y_column]].dropna().copy()
            if plot_df.empty:
                return None
            bins = max(8, min(30, int(len(plot_df) ** 0.5)))
            counts, bin_edges = pd.cut(plot_df[y_column], bins=bins, retbins=True)
            hist_counts = plot_df[y_column].groupby(counts).count().tolist()
            bin_labels = [f"{round(bin_edges[i], 2)}Ã¢â‚¬â€œ{round(bin_edges[i+1], 2)}" for i in range(len(bin_edges) - 1)]
            chart_title = _ensure_chart_title(title, question, "histogram", y=y_column)
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "category", "data": bin_labels, "axisLabel": {"rotate": 35, "color": "#c5c8d3", "fontSize": 10}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"show": False},
                "series": [{
                    "name": prettify_label(y_column),
                    "type": "bar",
                    "data": hist_counts,
                    "barCategoryGap": "2%",
                    "itemStyle": {"borderRadius": [3, 3, 0, 0]},
                }],
            }
            return _echarts_payload(option, "histogram", chart_title)

        # —— COMBO (column + line hybrid) ——————————————————————————————————————
        if chart_type == "combo":
            if not x_column:
                x_column = time_cols[0] if time_cols else (category_column or (df.columns[0] if len(df.columns) else None))
            if not y_columns:
                y_columns = _infer_line_series(df, question, requested=([y_column] if y_column else []))
            if not x_column or not y_columns or len(y_columns) < 2:
                # Need at least 2 measures for a meaningful combo chart
                # Fall back: if only 1 measure, duplicate logic won't help — return None
                if y_columns and len(y_columns) == 1:
                    # Single measure: render as bar chart instead
                    return render_chart_from_spec(df, question, "bar", x_column=x_column,
                                                  y_column=y_columns[0], category_column=category_column, title=title)
                return None

            plot_df = df[[x_column] + y_columns].dropna(subset=[x_column]).copy()
            if plot_df.empty:
                return None
            if pd.api.types.is_datetime64_any_dtype(plot_df[x_column]):
                agg_df = plot_df.groupby(x_column, as_index=False)[y_columns].sum(numeric_only=True).sort_values(x_column)
            else:
                plot_df[x_column] = plot_df[x_column].astype(str)
                agg_df = plot_df.groupby(x_column, as_index=False)[y_columns].sum(numeric_only=True).sort_values(x_column)
            if len(agg_df) < 2:
                return None

            chart_title = _ensure_chart_title(title, question, "combo", x=x_column, y=y_columns[0], metrics=y_columns)
            x_data = agg_df[x_column].astype(str).tolist()

            # First measure(s) as columns, remaining as lines
            # Default split: first column is bar, rest are lines
            bar_columns = y_columns[:1]
            line_columns = y_columns[1:]

            series = []
            # Bar series (primary y-axis, index 0)
            for i, bc in enumerate(bar_columns):
                series.append({
                    "name": prettify_label(bc),
                    "type": "bar",
                    "yAxisIndex": 0,
                    "data": [round(float(v), 4) if v == v else 0 for v in agg_df[bc].fillna(0).tolist()],
                    "itemStyle": {"borderRadius": [4, 4, 0, 0], "opacity": 0.85},
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}},
                })

            # Line series (secondary y-axis, index 1)
            for i, lc in enumerate(line_columns):
                series.append({
                    "name": prettify_label(lc),
                    "type": "line",
                    "yAxisIndex": 1,
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 7,
                    "lineStyle": {"width": 2.5},
                    "data": [round(float(v), 4) if v == v else None for v in agg_df[lc].tolist()],
                })

            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
                "xAxis": {
                    "type": "category",
                    "data": x_data,
                    "axisLabel": {"rotate": 30, "color": "#c5c8d3"},
                    "axisPointer": {"type": "shadow"},
                },
                "yAxis": [
                    {
                        "type": "value",
                        "name": prettify_label(bar_columns[0]),
                        "nameTextStyle": {"color": "#c5c8d3"},
                        "axisLabel": {"color": "#c5c8d3"},
                        "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
                    },
                    {
                        "type": "value",
                        "name": prettify_label(line_columns[0]) if line_columns else "",
                        "nameTextStyle": {"color": "#c5c8d3"},
                        "axisLabel": {"color": "#c5c8d3"},
                        "splitLine": {"show": False},
                    },
                ],
                "legend": {"data": [s["name"] for s in series]},
                "series": series,
            }
            return _echarts_payload(option, "combo", chart_title)

        # —— RADAR —————————————————————————————————────────────────————————————
        if chart_type == "radar":
            if not category_column:
                category_column = choose_category_column(df, measure_cols, time_cols)
            if not y_columns:
                y_columns = measure_cols[:5] # Up to 5 metrics
            if not category_column or not y_columns:
                return None
            agg_df = df[[category_column] + y_columns].dropna(subset=[category_column]).groupby(category_column, as_index=False)[y_columns].sum(numeric_only=True)
            if len(agg_df) > max_categories:
                agg_df = agg_df.head(max_categories)
            if agg_df.empty:
                return None
            
            # Setup indicators
            indicators = []
            for yc in y_columns:
                max_val = float(agg_df[yc].max())
                if max_val <= 0:
                    max_val = 100.0
                indicators.append({
                    "name": prettify_label(yc),
                    "max": round(max_val * 1.15, 2) # Buffer
                })

            radar_data = []
            for _, row in agg_df.iterrows():
                val_list = [round(float(row[yc]), 4) if row[yc] == row[yc] else 0 for yc in y_columns]
                radar_data.append({
                    "name": str(row[category_column]),
                    "value": val_list
                })

            chart_title = _ensure_chart_title(title, question, "radar", category=category_column)
            option = {
                "legend": {"data": [d["name"] for d in radar_data]},
                "radar": {
                    "indicator": indicators,
                    "splitArea": {
                        "show": True,
                        "areaStyle": {"color": ["rgba(255,255,255,0.01)", "rgba(255,255,255,0.03)"]}
                    },
                    "axisName": {"color": "#c5c8d3"}
                },
                "series": [{
                    "type": "radar",
                    "data": radar_data,
                    "symbol": "circle",
                    "symbolSize": 6
                }]
            }
            return _echarts_payload(option, "radar", chart_title)

        # —— FUNNEL —————————————————————————————————───────────────────────────
        if chart_type == "funnel":
            if not category_column:
                category_column = choose_category_column(df, measure_cols, time_cols)
            if not y_column:
                y_column = measure_cols[0] if measure_cols else None
            if not category_column or not y_column:
                return None
            agg_df = df[[category_column, y_column]].dropna(subset=[category_column, y_column]).groupby(category_column, as_index=False)[y_column].sum(numeric_only=True).sort_values(y_column, ascending=False)
            if agg_df.empty:
                return None
            if len(agg_df) > max_categories:
                agg_df = agg_df.head(max_categories)
            
            funnel_data = [
                {"name": str(row[category_column]), "value": round(float(row[y_column]), 4)}
                for _, row in agg_df.iterrows()
            ]
            chart_title = _ensure_chart_title(title, question, "funnel", y=y_column, category=category_column)
            option = {
                "legend": {"data": [d["name"] for d in funnel_data]},
                "series": [{
                    "name": prettify_label(y_column),
                    "type": "funnel",
                    "left": "10%",
                    "top": "15%",
                    "bottom": "10%",
                    "width": "80%",
                    "min": 0,
                    "max": float(agg_df[y_column].max() or 100),
                    "minSize": "0%",
                    "maxSize": "100%",
                    "sort": "descending",
                    "gap": 2,
                    "label": {"show": True, "position": "inside", "formatter": "{b}: {c}"},
                    "labelLine": {"show": False},
                    "itemStyle": {"borderColor": "#343541", "borderWidth": 1},
                    "emphasis": {"label": {"fontSize": 14}},
                    "data": funnel_data
                }]
            }
            return _echarts_payload(option, "funnel", chart_title)

        # —— GAUGE —————————————————————————————————————————————————————————————
        if chart_type == "gauge":
            if not y_column:
                y_column = measure_cols[0] if measure_cols else None
            if not y_column:
                return None
            val = float(df[y_column].sum(numeric_only=True))
            if val != val: # nan check
                val = 0.0
            
            # Auto target boundary: round up to nearest sensible order of magnitude
            import math
            if val > 0:
                magnitude = 10 ** math.floor(math.log10(val))
                max_bound = math.ceil(val / magnitude) * magnitude
                if max_bound == val:
                    max_bound = val * 1.25
            else:
                max_bound = 100.0

            chart_title = _ensure_chart_title(title, question, "gauge", y=y_column)
            option = {
                "series": [{
                    "type": "gauge",
                    "min": 0,
                    "max": round(max_bound, 2),
                    "progress": {"show": True, "width": 14},
                    "axisLine": {"lineStyle": {"width": 14, "color": [[0.3, "#26c6da"], [0.7, "#26a69a"], [1, "#2e7d32"]]}},
                    "pointer": {"itemStyle": {"color": "auto"}},
                    "axisTick": {"distance": -14, "splitNumber": 5, "lineStyle": {"width": 2, "color": "#999"}},
                    "splitLine": {"distance": -20, "length": 14, "lineStyle": {"width": 3, "color": "#999"}},
                    "axisLabel": {"distance": -20, "color": "#999", "fontSize": 12},
                    "anchor": {"show": False},
                    "title": {"show": True, "offsetCenter": [0, "70%"], "color": "#c5c8d3", "fontSize": 14},
                    "detail": {"valueAnimation": True, "offsetCenter": [0, "30%"], "color": "#ffffff", "fontSize": 24, "formatter": "{value}"},
                    "data": [{"value": round(val, 2), "name": prettify_label(y_column)}]
                }]
            }
            return _echarts_payload(option, "gauge", chart_title)

        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Named chart builders (used by AI tool dispatch)
# ---------------------------------------------------------------------------
def build_line_chart(dataset_id, x_column="", y_columns=None, category_column="", title=""):

    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="line", x_column=x_column or None, y_columns=y_columns or None, category_column=category_column or None, title=title or None) or {"error": "Unable to build line chart"}


def build_bar_chart(dataset_id, category_column="", y_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="bar", category_column=category_column or None, y_column=y_column or None, title=title or None) or {"error": "Unable to build bar chart"}


def build_clustered_bar_chart(dataset_id, category_column="", y_columns=None, title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="clustered_bar", category_column=category_column or None, y_columns=y_columns or None, title=title or None) or {"error": "Unable to build clustered bar chart"}


def build_pie_chart(dataset_id, category_column="", y_column="", donut=False, title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="donut" if donut else "pie", category_column=category_column or None, y_column=y_column or None, title=title or None) or {"error": "Unable to build pie chart"}


def build_scatter_chart(dataset_id, x_column="", y_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="scatter", x_column=x_column or None, y_column=y_column or None, title=title or None) or {"error": "Unable to build scatter chart"}


def build_histogram_chart(dataset_id, y_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="histogram", y_column=y_column or None, title=title or None) or {"error": "Unable to build histogram chart"}


def build_combo_chart(dataset_id, x_column="", bar_columns=None, line_columns=None, title=""):
    """Build a hybrid combo chart with column bars and line overlays on dual axes."""
    df = get_chart_context(dataset_id)
    # Merge bar and line columns into y_columns; render_chart_from_spec
    # will use the first as bar and the rest as lines for combo type.
    y_cols = []
    if bar_columns:
        y_cols.extend(bar_columns if isinstance(bar_columns, list) else [bar_columns])
    if line_columns:
        y_cols.extend(line_columns if isinstance(line_columns, list) else [line_columns])
    return render_chart_from_spec(
        df, question=title, chart_type="combo",
        x_column=x_column or None, y_columns=y_cols or None, title=title or None,
    ) or {"error": "Unable to build combo chart"}


def build_area_chart(dataset_id, x_column="", y_columns=None, category_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="area", x_column=x_column or None, y_columns=y_columns or None, category_column=category_column or None, title=title or None) or {"error": "Unable to build area chart"}


def build_stacked_bar_chart(dataset_id, category_column="", y_columns=None, title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="stacked_bar", category_column=category_column or None, y_columns=y_columns or None, title=title or None) or {"error": "Unable to build stacked bar chart"}


def build_radar_chart(dataset_id, category_column="", y_columns=None, title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="radar", category_column=category_column or None, y_columns=y_columns or None, title=title or None) or {"error": "Unable to build radar chart"}


def build_funnel_chart(dataset_id, category_column="", y_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="funnel", category_column=category_column or None, y_column=y_column or None, title=title or None) or {"error": "Unable to build funnel chart"}


def build_gauge_chart(dataset_id, y_column="", title=""):
    df = get_chart_context(dataset_id)
    return render_chart_from_spec(df, question=title, chart_type="gauge", y_column=y_column or None, title=title or None) or {"error": "Unable to build gauge chart"}


# ---------------------------------------------------------------------------
# AI Tool Definitions
# ---------------------------------------------------------------------------
CHART_TOOL_DEFINITIONS = [
    {
        "name": "build_line_chart",
        "description": "Build a line chart for a trend or time series view. Use category_column when the data has a grouping dimension (e.g. Region, Country, Product) so that each group gets its own line.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Dataset context id."},
                "x_column": {"type": "string", "description": "The time or ordered x-axis column (e.g. SalesMonth, Date)."},
                "y_columns": {"type": "array", "description": "One or more numeric measure columns to plot.", "items": {"type": "string"}},
                "category_column": {"type": "string", "description": "Optional grouping column (e.g. Region, Country). If provided, each unique value gets its own line."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_bar_chart",
        "description": "Build a horizontal bar chart for categorical comparisons with one measure.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_clustered_bar_chart",
        "description": "Build a clustered bar chart to compare multiple numeric measures across the same category.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_pie_chart",
        "description": "Build a pie or donut chart for part-to-whole composition.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_column": {"type": "string"},
                "donut": {"type": "boolean"},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_scatter_chart",
        "description": "Build a scatter chart for relationships between two numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "x_column": {"type": "string"},
                "y_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_histogram_chart",
        "description": "Build a histogram for the distribution of a numeric column.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "y_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_combo_chart",
        "description": "Build a hybrid combo chart that combines column bars (primary axis) with line overlays (secondary axis). Ideal for showing volume vs rate, revenue vs margin, quantity vs price, or any two related but differently-scaled metrics. First columns in bar_columns appear as bars, line_columns appear as smooth lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Dataset context id."},
                "x_column": {"type": "string", "description": "The x-axis column (time, category, etc.)."},
                "bar_columns": {"type": "array", "description": "Numeric columns to render as vertical bars (primary y-axis).", "items": {"type": "string"}},
                "line_columns": {"type": "array", "description": "Numeric columns to render as lines (secondary y-axis).", "items": {"type": "string"}},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_area_chart",
        "description": "Build an area chart (filled line chart) showing trend over time or categories. Used for volume/accumulation trends over time.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Dataset context id."},
                "x_column": {"type": "string", "description": "The time or category x-axis column."},
                "y_columns": {"type": "array", "description": "Numeric columns to plot.", "items": {"type": "string"}},
                "category_column": {"type": "string", "description": "Optional grouping column."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_stacked_bar_chart",
        "description": "Build a stacked bar chart showing segment breakdowns inside category totals (e.g. Sales stacked by Product Category).",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_radar_chart",
        "description": "Build a radar chart showing multi-axis category profiles. Ideal for comparing multi-dimensional category details.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_columns": {"type": "array", "description": "Up to 5 numeric columns representing the radar axes.", "items": {"type": "string"}},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_funnel_chart",
        "description": "Build a funnel chart showing progressive dropdown steps or conversions in stages (e.g. Lead -> Deal -> Win).",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "category_column": {"type": "string"},
                "y_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "build_gauge_chart",
        "description": "Build a gauge chart representing a single numeric metric relative to targets or scale bounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "y_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
    },
]

CHART_TOOL_MAP = {
    "build_line_chart": build_line_chart,
    "build_bar_chart": build_bar_chart,
    "build_clustered_bar_chart": build_clustered_bar_chart,
    "build_pie_chart": build_pie_chart,
    "build_scatter_chart": build_scatter_chart,
    "build_histogram_chart": build_histogram_chart,
    "build_combo_chart": build_combo_chart,
    "build_area_chart": build_area_chart,
    "build_stacked_bar_chart": build_stacked_bar_chart,
    "build_radar_chart": build_radar_chart,
    "build_funnel_chart": build_funnel_chart,
    "build_gauge_chart": build_gauge_chart,
}


def build_chart_context_summary(df, question=None):
    if df is None or df.empty:
        return {"row_count": 0, "columns": [], "sample_rows": []}
    parsed = try_parse_dates(df.copy())
    numeric_cols = choose_measure_columns(parsed)
    time_cols = [c for c in parsed.columns if pd.api.types.is_datetime64_any_dtype(parsed[c])]
    category_cols = [
        c for c in parsed.columns
        if c not in numeric_cols and c not in time_cols and parsed[c].nunique(dropna=True) > 1
    ]
    sample_df = parsed.head(5).copy()
    for col in sample_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "question": question or "",
        "row_count": int(len(parsed)),
        "columns": [
            {
                "name": str(col),
                "dtype": str(parsed[col].dtype),
                "non_null_ratio": round(float(parsed[col].notna().mean()), 3),
                "unique_values": int(parsed[col].nunique(dropna=True)),
            }
            for col in parsed.columns
        ],
        "numeric_candidates": numeric_cols[:8],
        "time_candidates": time_cols[:8],
        "category_candidates": category_cols[:8],
        "sample_rows": sample_df.fillna("").astype(str).to_dict(orient="records"),
    }


def build_chart_with_tools(user_query, df, sql_query=None):
    if df is None or df.empty:
        return None

    dataset_id = register_chart_context(df)
    if _should_force_clustered_bar(user_query, df):
        measure_cols = choose_measure_columns(df)
        time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        category_column = choose_category_column(df, measure_cols, time_cols)
        y_columns = _infer_clustered_bar_series(df, user_query)
        if category_column and len(y_columns) >= 2:
            forced_chart = build_clustered_bar_chart(
                dataset_id=dataset_id,
                category_column=category_column,
                y_columns=y_columns,
                title="",
            )
            if isinstance(forced_chart, dict) and forced_chart.get("data"):
                clear_chart_context(dataset_id)
                return forced_chart

    chart_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.2,
    ).bind_tools(CHART_TOOL_DEFINITIONS, tool_choice="auto")

    summary = build_chart_context_summary(df, question=user_query)
    system_prompt = """
You are a chart selection assistant for a data chat application.
Pick the single best chart tool for the user's question and dataset.

Rules:
- Use a line chart for time series or ordered trends.
- Use multiple lines on a line chart when the question compares two or more metrics over the same time axis.
- Use a bar chart for category comparisons.
- Use a clustered bar chart when comparing multiple measures across the same category.
- Use a pie or donut chart only for part-to-whole composition with a small number of categories.
- Use a scatter chart for relationships between two numeric columns.
- Use a histogram for distribution of one numeric column.
- Example: "How does profit per country compare to sales quantity?" should use a clustered bar chart.
- Example: "Compare revenue and profit over time" should use a line chart with two series.
- Prefer the simplest chart that answers the question.
- If a chart is not useful, do not call a tool.
- When calling a tool, pass the dataset_id and the most relevant columns from the summary.

Dataset summary:
{summary}

SQL query:
{sql_query}
"""

    try:
        result = chart_model.invoke([
            SystemMessage(content=system_prompt.format(
                summary=json.dumps(summary, ensure_ascii=True, default=str),
                sql_query=sql_query or "",
            )),
            HumanMessage(content=user_query or ""),
        ])
        tool_calls = getattr(result, "tool_calls", None) or []
        if not tool_calls:
            return generate_chart_base64(df, user_query)

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {}) or {}
        if tool_name not in CHART_TOOL_MAP:
            return generate_chart_base64(df, user_query)

        tool_args = dict(tool_args)
        tool_args["dataset_id"] = dataset_id
        if "title" in tool_args and isinstance(tool_args["title"], str):
            tool_args["title"] = tool_args["title"].strip()
            if tool_args["title"] and re.search(r"\b(how|what|why|compare|compare[s]?)\b", tool_args["title"].lower()):
                tool_args["title"] = ""

        if tool_name == "build_bar_chart":
            inferred_clustered = _infer_clustered_bar_series(df, user_query, requested=tool_args.get("y_columns"))
            if len(inferred_clustered) >= 2:
                tool_name = "build_clustered_bar_chart"
                tool_args.pop("y_column", None)
                tool_args["y_columns"] = inferred_clustered
                if not tool_args.get("category_column"):
                    measure_cols = choose_measure_columns(df)
                    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    tool_args["category_column"] = choose_category_column(df, measure_cols, time_cols)
        elif tool_name == "build_clustered_bar_chart":
            tool_args["y_columns"] = _infer_clustered_bar_series(df, user_query, requested=tool_args.get("y_columns"))

        chart_payload = CHART_TOOL_MAP[tool_name](**tool_args)
        if isinstance(chart_payload, dict) and (chart_payload.get("option") or chart_payload.get("data")):
            return chart_payload
        if isinstance(chart_payload, str):
            try:
                parsed = json.loads(chart_payload)
                if parsed.get("option") or parsed.get("data"):
                    return parsed
            except Exception:
                pass
        return generate_chart_base64(df, user_query)

    except Exception:
        return generate_chart_base64(df, user_query)
    finally:
        clear_chart_context(dataset_id)


def should_generate_chart(question, df):
    if df is None or df.empty:
        return False
    df = try_parse_dates(df.copy())
    if len(df) < 2:
        return False
    measure_cols = choose_measure_columns(df)
    if not measure_cols:
        return False
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    category_col = choose_category_column(df, measure_cols, time_cols)
    return bool(time_cols or category_col or len(measure_cols) >= 2)


def has_visual_intent(question):
    q = (question or "").lower()

    # Normalize common compound chart words to their spaced equivalents
    q = re.sub(r'\bbar[\s-]?graph\b', 'bar chart', q)
    q = re.sub(r'\bline[\s-]?graph\b', 'line chart', q)
    q = re.sub(r'\bpie[\s-]?graph\b', 'pie chart', q)
    q = re.sub(r'\bbar[\s-]?chart\b', 'bar chart', q)
    q = re.sub(r'\bpie[\s-]?chart\b', 'pie chart', q)
    q = re.sub(r'\bline[\s-]?chart\b', 'line chart', q)
    q = re.sub(r'\bdonut[\s-]?chart\b', 'donut chart', q)
    q = re.sub(r'\bscatter[\s-]?plot\b', 'scatter chart', q)

    patterns = [
        # action verb + chart type (handles "generate a bar chart", "show me a pie chart")
        r"\b(generate|create|show|make|draw|plot|render|give me|produce|display|come up with|build|visualize)\b.{0,50}\b(chart|graph|visual|visualization|plot|diagram|bar|pie|donut|line|scatter|histogram)\b",
        # chart type + "for/of/this" (handles "bar chart for sales", "pie chart of returns")
        r"\b(chart|graph|visual|visualization|plot|diagram|bar chart|pie chart|line chart|donut chart|scatter chart)\b.{0,40}\b(for|of|this|the|above|these|on|by)\b",
        # pure visualize/visualization keyword
        r"\b(visuali[sz]e|visuali[sz]ation)\b",
        # "show it/this as a chart"
        r"\bshow (it |this |the data |that )?as (a |an )?(chart|graph|visual|plot|bar|pie|line|scatter)\b",
        # "draw/plot this"
        r"\b(can you |please )?(draw|plot|chart|graph) (this|it|the (data|result))\b",
        # standalone generate a chart
        r"\bgenerate (a |an )?(visual|chart|graph|plot|bar|pie|line|scatter|histogram)\b",
    ]
    return any(re.search(p, q) for p in patterns)



def generate_chart_base64(df, question, chart_pref=None):
    """Fallback chart builder â€” auto-selects the best chart type and returns an ECharts option."""
    if df is None or df.empty:
        return None
    try:
        df = df.copy()
        df = try_parse_dates(df)
        measure_cols = choose_measure_columns(df)
        time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        category_col = choose_category_column(df, measure_cols, time_cols)

        if not measure_cols:
            return None

        chart_type = None
        x_field = None
        y_field = None
        metric_fields = None
        category_field = None
        option = {}

        if time_cols:
            x = time_cols[0]
            top_measures = measure_cols[:3]
            plot_df = df[[x] + top_measures].dropna(subset=[x]).copy()
            if plot_df.empty:
                return None
            agg_df = plot_df.groupby(x, as_index=False)[top_measures].sum(numeric_only=True).sort_values(x)
            if len(agg_df) < 2:
                return None
            chart_type = "line"
            x_field = x
            metric_fields = top_measures
            x_data = agg_df[x].astype(str).tolist()
            series = []
            for m in top_measures:
                series.append({
                    "name": prettify_label(m),
                    "type": "line",
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 6,
                    "data": [round(float(v), 4) if v == v else None for v in agg_df[m].tolist()],
                })
            option = {
                "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30, "color": "#c5c8d3"}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"data": [s["name"] for s in series]} if len(series) > 1 else {"show": False},
                "series": series,
            }

        elif category_col:
            y = measure_cols[0]
            agg_df = (
                df[[category_col, y]]
                .dropna(subset=[category_col, y])
                .groupby(category_col, as_index=False)[y]
                .sum(numeric_only=True)
                .sort_values(y, ascending=False)
            )
            if agg_df.empty:
                return None
            if len(agg_df) > 12:
                agg_df = agg_df.head(12)
            chart_type = "bar"
            y_field = y
            category_field = category_col
            cat_data = agg_df[category_col].astype(str).tolist()[::-1]
            val_data = [round(float(v), 4) if v == v else 0 for v in agg_df[y].fillna(0).tolist()][::-1]
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "yAxis": {"type": "category", "data": cat_data, "axisLabel": {"color": "#c5c8d3"}},
                "grid": {"left": "3%", "right": "4%", "bottom": "4%", "top": "8%", "containLabel": True},
                "legend": {"show": False},
                "series": [{
                    "name": prettify_label(y),
                    "type": "bar",
                    "data": val_data,
                    "itemStyle": {"borderRadius": [0, 4, 4, 0]},
                }],
            }

        elif len(measure_cols) >= 2:
            x = measure_cols[0]
            y = measure_cols[1]
            plot_df = df[[x, y]].dropna().copy()
            if plot_df.empty:
                return None
            if len(plot_df) > 2000:
                plot_df = plot_df.sample(2000, random_state=42)
            chart_type = "scatter"
            x_field = x
            y_field = y
            scatter_data = [[round(float(r[x]), 4), round(float(r[y]), 4)] for _, r in plot_df.iterrows()]
            option = {
                "xAxis": {"type": "value", "name": prettify_label(x), "nameLocation": "middle", "nameGap": 30, "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "yAxis": {"type": "value", "name": prettify_label(y), "nameLocation": "middle", "nameGap": 40, "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"show": False},
                "series": [{"name": f"{prettify_label(y)} vs {prettify_label(x)}", "type": "scatter", "data": scatter_data, "symbolSize": 6, "itemStyle": {"opacity": 0.75}}],
            }

        else:
            y = measure_cols[0]
            plot_df = df[[y]].dropna().copy()
            if plot_df.empty:
                return None
            bins = max(8, min(30, int(len(plot_df) ** 0.5)))
            counts, bin_edges = pd.cut(plot_df[y], bins=bins, retbins=True)
            hist_counts = plot_df[y].groupby(counts).count().tolist()
            bin_labels = [f"{round(bin_edges[i], 2)}â€“{round(bin_edges[i+1], 2)}" for i in range(len(bin_edges) - 1)]
            chart_type = "histogram"
            y_field = y
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "category", "data": bin_labels, "axisLabel": {"rotate": 35, "color": "#c5c8d3", "fontSize": 10}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#c5c8d3"}, "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
                "legend": {"show": False},
                "series": [{"name": prettify_label(y), "type": "bar", "data": hist_counts, "barCategoryGap": "2%", "itemStyle": {"borderRadius": [3, 3, 0, 0]}}],
            }

        if not option:
            return None

        chart_title = build_chart_title(question=question, chart_type=chart_type, x=x_field, y=y_field, metrics=metric_fields, category=category_field)
        return _echarts_payload(option, chart_type, chart_title)

    except Exception:
        return None

        return None
