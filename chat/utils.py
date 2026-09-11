import os
import datetime
import uuid
import re
import json
import requests


import pandas as pd

# Suppress ONNXRuntime CUDA warning for ChromaDB (uses CPU for local message embedding)
try:
    import onnxruntime as ort
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
except Exception:
    pass

import chromadb
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# LLM Factory (Ollama — local inference)
# ---------------------------------------------------------------------------
def _make_ollama(model_env_key: str, temperature: float, num_ctx: int = None) -> ChatOllama:
    """Build a ChatOllama instance using shared Ollama env settings."""
    default_model = "pxlksr/defog_sqlcoder-7b-2:Q8" if "SQL" in model_env_key else "qwen2.5:7b"
    model = (os.getenv(model_env_key) or default_model).strip()
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
    keep_alive = (os.getenv("OLLAMA_KEEP_ALIVE") or "24h").strip()
    ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    return ChatOllama(
        model=model,
        base_url=host,
        temperature=temperature,
        keep_alive=keep_alive,
        num_ctx=ctx,
    )


def get_sql_model(temperature: float = 0.0) -> ChatOllama:
    """
    LLM for SQL generation and self-healing (reads OLLAMA_SQL_MODEL).
    Low temperature keeps SQL output deterministic.
    """
    return _make_ollama("OLLAMA_SQL_MODEL", temperature)


def get_nlp_model(temperature: float = 0.4) -> ChatOllama:
    """
    LLM for intent classification and human-readable response formatting
    (reads OLLAMA_NLP_MODEL).
    """
    return _make_ollama("OLLAMA_NLP_MODEL", temperature)


def get_chat_model(temperature: float = 0.4) -> ChatOllama:
    """
    Generic factory used by chart-tool builder and other call sites.
    Delegates to get_nlp_model so it always honours OLLAMA_NLP_MODEL.
    """
    return get_nlp_model(temperature)


# ---------------------------------------------------------------------------
# Module-level singletons — instantiated once at import time
# ---------------------------------------------------------------------------
llm     = get_nlp_model(temperature=0.7)   # intent classification + response formatting
sql_llm = get_sql_model(temperature=0.0)   # SQL generation + self-healing

# ---------------------------------------------------------------------------
# ChromaDB (Stored in project directory on D: drive to avoid C: drive space exhaustion)
# ---------------------------------------------------------------------------
chroma_data_path = os.getenv(
    "CHROMA_DATA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
)
os.makedirs(chroma_data_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_data_path)

sessions_collection  = chroma_client.get_or_create_collection(name="chat_sessions")
messages_collection  = chroma_client.get_or_create_collection(name="chat_messages")
charts_collection    = chroma_client.get_or_create_collection(name="chat_charts")
feedback_collection  = chroma_client.get_or_create_collection(name="chat_feedback")

# ---------------------------------------------------------------------------
# In-memory caches & Pre-warmed NSSF Uganda Ledger Intro
# ---------------------------------------------------------------------------
NSSF_LEDGER_DEFAULT_INTRO = (
    "This chatbot provides intelligent financial analytics and conversational SQL querying across the National Social Security Fund "
    "(NSSF) Uganda General Ledger (staging.proddta_f0911_account_ledger). You can explore transactions, analyze operational expenditure, "
    "track vouchers and payments, review balance adjustments, and audit journal entries."
)

NSSF_LEDGER_DEFAULT_ANALYSIS = (
    "You can run trend analysis, period comparisons (MoM/YoY/QoQ), drill-downs by cost center/business unit (glmcu), natural account (globj), "
    "subledger (glsbl), and document type (gldct), monitor expenditure vs budget, generate interactive charts, and export data."
)

NSSF_LEDGER_DEFAULT_SUGGESTIONS = [
    "What is the total expenditure in the ledger for fiscal year 2021?",
    "Show the top 10 cost centers by total expenditure.",
    "What is the monthly expenditure trend for fiscal year 2021?",
    "Show total voucher payments (PV) grouped by vendor or account.",
    "Compare total debits and credits for fiscal year 2020 vs 2021.",
    "Show the largest 10 journal transactions in the ledger.",
]

dataset_intro_cache = {
    "text": NSSF_LEDGER_DEFAULT_INTRO,
    "analysis": NSSF_LEDGER_DEFAULT_ANALYSIS,
    "suggestions": NSSF_LEDGER_DEFAULT_SUGGESTIONS,
    "updated_at": datetime.datetime.now(),
}
chart_context_store = {}


import urllib.parse
from langchain_community.utilities import SQLDatabase

# ---------------------------------------------------------------------------
_CACHED_SQL_DATABASE = None
_CACHED_TABLE_INFO = None   # schema string cached after first fetch


# ---------------------------------------------------------------------------
# Database (PostgreSQL - NSSF Uganda Data Warehouse)
# ---------------------------------------------------------------------------
def init_database(user=None, password=None, database=None, force_reconnect=False) -> SQLDatabase:
    global _CACHED_SQL_DATABASE

    if _CACHED_SQL_DATABASE is not None and not force_reconnect:
        return _CACHED_SQL_DATABASE

    # Direct PostgreSQL Warehouse (requires active VPN connection to NSSF Uganda network)
    user = user or os.getenv("DB_USER")
    password = password or os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "192.168.193.8")
    port = os.getenv("DB_PORT", "5432")
    database = database or os.getenv("DB_NAME", "fund_warehouse_db")
    
    if not user or not password:
        raise ValueError("Missing database credentials! Please configure your .env file.")
    
    # URL-encode the password to handle any special characters safely
    safe_password = urllib.parse.quote_plus(password)
    db_uri = f"postgresql+psycopg2://{user}:{safe_password}@{host}:{port}/{database}"
    global _CACHED_TABLE_INFO
    _CACHED_TABLE_INFO = None  # reset if reconnecting
    _CACHED_SQL_DATABASE = SQLDatabase.from_uri(
        db_uri,
        schema="staging",
        include_tables=["proddta_f0911_account_ledger"],
        sample_rows_in_table_info=0,
        view_support=True,
        engine_args={"connect_args": {"connect_timeout": 5}}
    )
    # Pre-fetch schema eagerly so the first query doesn't pay the cost
    try:
        _CACHED_TABLE_INFO = _CACHED_SQL_DATABASE.get_table_info()
    except Exception:
        pass
    return _CACHED_SQL_DATABASE

# ---------------------------------------------------------------------------
# Currency Conversion
# ---------------------------------------------------------------------------
exchange_rate_cache = {
    "rates_text": "",
    "updated_at": None
}

def get_exchange_rates_context(question: str = ""):
    # Only fetch and inject exchange rates if the user actually asks for forex/currency conversion
    if question:
        forex_keywords = ["usd", "eur", "gbp", "exchange rate", "forex", "fx", "conversion", "convert", "dollar", "currency", "shilling"]
        if not any(kw in question.lower() for kw in forex_keywords):
            return ""

    now = datetime.datetime.now()
    if exchange_rate_cache["rates_text"] and exchange_rate_cache["updated_at"]:
        if (now - exchange_rate_cache["updated_at"]).total_seconds() < 3600 * 12:  # 12 hours
            return exchange_rate_cache["rates_text"]

    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=4)
        if response.status_code == 200:
            data = response.json()
            rates = data.get("rates", {})
            major_currencies = ["EUR", "GBP", "UGX", "KES", "RWF", "USD"]
            lines = [f"1 USD = {rates[c]:,.2f} {c}" for c in major_currencies if c in rates]
            rates_text = "Live Exchange Rates (Base 1 USD):\n" + "\n".join(lines)
            exchange_rate_cache["rates_text"] = rates_text
            exchange_rate_cache["updated_at"] = now
            return rates_text
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Financial & Accounting Domain Glossary Context
# ---------------------------------------------------------------------------
finance_glossary_cache = {"text": None}

def get_finance_glossary_context():
    """
    Loads and caches the official NSSF Uganda Financial & Accounting Domain Glossary.
    """
    if finance_glossary_cache["text"] is not None:
        return finance_glossary_cache["text"]

    glossary_path = os.path.join(os.path.dirname(__file__), "glossary_context.txt")
    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    finance_glossary_cache["text"] = content
                    return content
        except Exception:
            pass
    return ""


def get_targeted_glossary_context(question: str = "") -> str:
    """
    Returns targeted glossary definitions based on terms present in the user question.
    Significantly minimizes prompt token count and maximizes Time-to-First-Token (TTFT).
    """
    full_glossary = get_finance_glossary_context()
    if not full_glossary:
        return ""

    if not question:
        return "NSSF Uganda GL Rules: Amounts in UGX. Credits (glaa < 0) = Revenue. Debits (glaa > 0) = Expenditure. 6-digit codes (globj) = Natural Accounts; 3-digit codes (glmcu) = Cost Centers."

    q_lower = question.lower()
    lines = [line.strip() for line in full_glossary.split("\n") if line.strip().startswith("*")]

    matched_lines = []
    for line in lines:
        term_match = re.match(r"\*\s*([^(:]+)(?:\(([^)]+)\))?\s*:", line)
        if term_match:
            main_term = term_match.group(1).strip().lower()
            alias = (term_match.group(2) or "").strip().lower()
            if (main_term and re.search(rf"\b{re.escape(main_term)}\b", q_lower)) or (alias and re.search(rf"\b{re.escape(alias)}\b", q_lower)):
                matched_lines.append(line)

    if matched_lines:
        return "[RELEVANT NSSF FINANCIAL GLOSSARY DEFINITIONS]\n" + "\n".join(matched_lines[:6])

    return "NSSF Uganda GL Context: Amounts in UGX. Credits (glaa < 0) = Revenue. Debits (glaa > 0) = Expenses. 6-digit codes (globj) = Natural Accounts; 3-digit codes (glmcu) = Cost Centers."


# ---------------------------------------------------------------------------
# LLM Chains
# ---------------------------------------------------------------------------
def get_sql_chain(db):
    template = """You are an expert PostgreSQL Financial Data Engineer for NSSF Uganda.
Generate a valid, precise, and executable PostgreSQL query to answer the user's question from table `staging.proddta_f0911_account_ledger`.

TABLE & KEY COLUMNS:
Table: staging.proddta_f0911_account_ledger
- glaa (numeric): Transaction Amount in Uganda Shillings (UGX).
  * GENERAL TRANSACTION AMOUNT / VOLUME: `SUM(ABS(glaa)) AS total_amount_ugx`
  * REVENUE / COLLECTIONS / INFLOWS: Credits are NEGATIVE amounts (`glaa < 0`). Calculate revenue with: `SUM(CASE WHEN glaa < 0 THEN ABS(glaa) ELSE 0 END) AS revenue_ugx` (or `WHERE glaa < 0` with `SUM(ABS(glaa))`).
  * EXPENDITURES / EXPENSES / OUTFLOWS: Debits are POSITIVE amounts (`glaa > 0`). Calculate expenditure with: `SUM(CASE WHEN glaa > 0 THEN glaa ELSE 0 END) AS expense_ugx` (or `WHERE glaa > 0` with `SUM(glaa)`).
  * NET FINANCIAL BALANCE: `SUM(glaa) AS net_balance_ugx`.
  * TOTAL TURNOVER / VOLUME: `SUM(ABS(glaa)) AS total_turnover_ugx`.
- gldgj (integer): Accounting Date (GL Date) stored as Julian CYDDD integer (e.g. 126252).
  * ALWAYS convert to date with: `TO_DATE((1900000 + gldgj)::text, 'YYYYDDD')`
  * Exact Date filter (e.g. "9th September 2026", "2026-09-09"): `TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') = '2026-09-09'`
  * Date Range / Month (e.g. "September 2026"): `TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') BETWEEN '2026-09-01' AND '2026-09-30'`
  * Calendar Year (e.g. "in 2026", "2025 revenue"): `TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') BETWEEN '2026-01-01' AND '2026-12-31'`
  * Monthly Trend / Breakdown: `SELECT TO_CHAR(TO_DATE((1900000 + gldgj)::text, 'YYYYDDD'), 'YYYY-MM') AS month, SUM(ABS(glaa)) AS revenue_ugx FROM staging.proddta_f0911_account_ledger WHERE ... GROUP BY 1 ORDER BY 1`
  * NEVER use EXTRACT() directly on Julian integer columns.
- glpost (text): Posting Status ('P'=Posted, blank=Unposted, 'D'=Deleted). ALWAYS filter `WHERE glpost = 'P'`.
- glmcu (text): Cost Center / Business Unit (e.g. '999' = Head Office).
- globj (text): 6-digit Natural Account / Object Account (e.g. '405014', '505102').
- gldct (text): Document Type ('PV'=Accounts Payable Voucher, 'JE'=Journal Entry, 'RI'=Invoice, 'PM'=Payment, '##'=Opening Balance).
- gldoc (numeric): Document / Voucher Number.
- glexa (text): Transaction Description (use `TRIM(glexa) ILIKE '%search%'`).
- gluser (text): Staff username who entered the transaction.
- glco (text): Company Code (e.g. '00001').
- glan8 (numeric): Address / Vendor / Employee / Member Number.

CRITICAL RULES:
1. Output ONLY the executable PostgreSQL SQL query. No explanations, markdown fences, or comments.
2. ALWAYS construct a valid SQL query for questions asking about transaction amounts, metrics, dates, or accounts.
3. FOLLOW-UPS & CONVERSATIONAL CONTEXT:
   If the question is a follow-up (e.g. "check the database", "give it", "show details", "what about in 2025", "break it down by cost center", "tell me more"), inspect the Conversation History to extract the previous metric, entity, or date and construct the appropriate SQL query.
4. Large Table Safety: The table has over 12 million rows. Always use aggregation (SUM, COUNT, AVG) with GROUP BY, or add `LIMIT 50` for transaction listings.
5. Suffixes / Semicolons: Semicolons at the end are allowed.

GOLD-STANDARD EXAMPLES:
- "how much was the transaction for 9th september 2026" / "transaction amount on 9th sep 2026":
  SELECT SUM(ABS(glaa)) AS total_amount_ugx FROM staging.proddta_f0911_account_ledger WHERE TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') = '2026-09-09' AND glpost = 'P';

- "what is the revenue of 9th september 2026" / "revenue on 9th sep 2026":
  SELECT SUM(ABS(glaa)) AS revenue_ugx FROM staging.proddta_f0911_account_ledger WHERE TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') = '2026-09-09' AND glpost = 'P' AND glaa < 0;

- "how many transactions were done on 9th september 2026":
  SELECT COUNT(*) AS transaction_count FROM staging.proddta_f0911_account_ledger WHERE TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') = '2026-09-09' AND glpost = 'P';

- "what is the revenue in 2026, monthly revenue":
  SELECT TO_CHAR(TO_DATE((1900000 + gldgj)::text, 'YYYYDDD'), 'YYYY-MM') AS month, SUM(ABS(glaa)) AS revenue_ugx FROM staging.proddta_f0911_account_ledger WHERE TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') BETWEEN '2026-01-01' AND '2026-12-31' AND glpost = 'P' AND glaa < 0 GROUP BY 1 ORDER BY 1;

- "total expenditure in 2021":
  SELECT SUM(glaa) AS total_expenditure_ugx FROM staging.proddta_f0911_account_ledger WHERE TO_DATE((1900000 + gldgj)::text, 'YYYYDDD') BETWEEN '2021-01-01' AND '2021-12-31' AND glpost = 'P' AND glaa > 0;

- "top 10 cost centers by total expenditure":
  SELECT TRIM(glmcu) AS cost_center, SUM(glaa) AS total_expenditure_ugx FROM staging.proddta_f0911_account_ledger WHERE glpost = 'P' AND glaa > 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 10;

- "how many accounts do we have there":
  SELECT COUNT(DISTINCT TRIM(globj)) AS total_natural_accounts, COUNT(DISTINCT TRIM(glmcu)) AS total_cost_centers FROM staging.proddta_f0911_account_ledger WHERE glpost = 'P';

Conversation History:
{chat_history}

Today's Date: {current_date}
User Question: {question}

Generate SQL:"""

    prompt = ChatPromptTemplate.from_template(template)

    def get_current_date(_):
        return datetime.datetime.now().strftime("%Y-%m-%d")

    return (
        RunnablePassthrough.assign(
            current_date=get_current_date,
        )
        | prompt
        | sql_llm
        | StrOutputParser()
    )


def build_no_sql_response(user_query, chat_history):
    template = """
        You are a helpful Senior Financial Analyst assistant for NSSF Uganda (National Social Security Fund).
        Provide a concise, executive, helpful response for general conversation or capabilities.

        ⛔ ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER output ANY SQL queries, SQL code blocks, statements, or programming code.
        - NEVER output markdown code fences or backticks.
        - If the user asks for specific amounts, dates, or transactions, DO NOT invent figures. Offer to query the General Ledger.

        GUIDELINES:
        - Explain relevant financial metrics, transactions, and cost centers tracked in the NSSF General Ledger.
        - Return TWO sections in this order:
          1) Response: direct, insightful answer
          2) Suggestive analysis: 3-5 short question-style follow-ups (one per line, under 14 words each)

        {finance_glossary}
        {currency_context}

        Question: {question}
        Conversation History: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query, 
        "chat_history": chat_history,
        "currency_context": get_exchange_rates_context(user_query),
        "finance_glossary": get_targeted_glossary_context(user_query),
    })


def stream_no_sql_response(user_query, chat_history):
    template = """
        You are a helpful Senior Financial Analyst assistant for NSSF Uganda (National Social Security Fund).
        Provide a concise, executive, helpful response for general conversation or capabilities.

        ⛔ ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER output ANY SQL queries, SQL code blocks, statements, or programming code.
        - NEVER output markdown code fences or backticks.
        - If the user asks for specific amounts, dates, or transactions, DO NOT invent figures. Offer to query the General Ledger.

        GUIDELINES:
        - Explain relevant financial metrics, transactions, and cost centers tracked in the NSSF General Ledger.
        - Return TWO sections in this order:
          1) Response: direct, insightful answer
          2) Suggestive analysis: 3-5 short question-style follow-ups (one per line, under 14 words each)

        {finance_glossary}
        {currency_context}

        Question: {question}
        Conversation History: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "question": user_query, 
        "chat_history": chat_history,
        "currency_context": get_exchange_rates_context(user_query),
        "finance_glossary": get_targeted_glossary_context(user_query),
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
        # Financial metrics & common typos (revebue, revenu, expens, etc.)
        r"\b(revebue|revenu|revenue|sales|income|inflow|collections?|contributions?)\b",
        r"\b(expens\w*|expenditur\w*|cost|spend\w*|spent|payout|disbursement)\b",
        r"\b(budget|balance|debit|credit|voucher|payment|invoice|turnover|net|ledger|allocation)\b",
        # Aggregations, metrics & questions
        r"\b(how\s+many|how\s+much|total|average|avg|sum|max|min|count|percent|growth|variance|difference)\b",
        r"\b(select|show\s+me|list|generate|plot|chart|graph|table|tabulate|breakdown|drill|compare|trend)\b",
        # Dates, months, days, years, quarters
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        r"\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\b",
        r"\b(from\s+20\d{2}|since\s+20\d{2}|between\s+20\d{2}|in\s+20\d{2}|fy\s*\d{2,4}|20\d{2})\b",
        r"\b(today|yesterday|this\s+month|last\s+month|current\s+month|this\s+year|last\s+year|ytd|mtd|q[1-4]|quarter)\b",
        # Explicit database actions & follow-ups
        r"\b(check\s+the\s+database|query\s+(the\s+)?database|in\s+(the\s+)?database|give\s+it|show\s+it|get\s+it|pull\s+it|run\s+it|tell\s+me\s+more|details)\b",
        # Entities & Ledger terms
        r"\b(transactions?|transacton\w*|entries|entry|journal|f0911|account\w*|acount\w*|cost\s*center\w*|object\s*account\w*|subledger)\b",
        r"\b(rows?|raws?|records?|lines|items?|size|dataset size|last record|latest date|newest|oldest|earliest)\b",
        r"\b(in (this|the) (table|database|ledger|dataset|schema))\b",
    ]
    if any(re.search(pattern, q) for pattern in sql_indicators):
        return True
    return False


def build_data_overview_response(user_query, db, chat_history):
    template = """
        You are the Senior Financial Data Analyst for NSSF Uganda (National Social Security Fund).
        The dataset is the NSSF Uganda General Ledger (`staging.proddta_f0911_account_ledger`).

        ⛔ ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER output ANY SQL queries, code blocks, or programming scripts.
        - NEVER output markdown code fences or backticks.

        Explain clearly in executive prose:
        - The domain: NSSF Uganda financial accounting, member contributions, investment revenues, and operational expenditures.
        - Key dimensions: GL date (`gldgj`), amounts in UGX (`glaa`), Cost Centers (`glmcu`, e.g. 999=Head Office), Natural Accounts (`globj`), Document Types (`gldct`), posting status (`glpost`).
        - Important metrics: Monthly revenues/collections (credits, glaa < 0), expenditures (debits, glaa > 0), budget variance, and cost center breakdowns.

        {finance_glossary}

        User Question: {question}
        Conversation History: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "finance_glossary": get_targeted_glossary_context(user_query),
    })


def stream_data_overview_response(user_query, db, chat_history):
    template = """
        You are the Senior Financial Data Analyst for NSSF Uganda (National Social Security Fund).
        The dataset is the NSSF Uganda General Ledger (`staging.proddta_f0911_account_ledger`).

        ⛔ ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER output ANY SQL queries, code blocks, or programming scripts.
        - NEVER output markdown code fences or backticks.

        Explain clearly in executive prose:
        - The domain: NSSF Uganda financial accounting, member contributions, investment revenues, and operational expenditures.
        - Key dimensions: GL date (`gldgj`), amounts in UGX (`glaa`), Cost Centers (`glmcu`, e.g. 999=Head Office), Natural Accounts (`globj`), Document Types (`gldct`), posting status (`glpost`).
        - Important metrics: Monthly revenues/collections (credits, glaa < 0), expenditures (debits, glaa > 0), budget variance, and cost center breakdowns.

        {finance_glossary}

        User Question: {question}
        Conversation History: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "question": user_query,
        "chat_history": chat_history,
        "finance_glossary": get_targeted_glossary_context(user_query),
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

    # Run automated financial calculator for rate changes, CAGR, or exchange rate variances
    rate_change_section = ""
    try:
        from chat.financial_tools import compute_rate_changes_from_sql_result
        rate_change_calculations = compute_rate_changes_from_sql_result(sql_result, user_query)
        if rate_change_calculations:
            rate_change_section = f"\n[VERIFIED FINANCIAL CALCULATIONS (CALCULATOR TOOL)]\n{rate_change_calculations}\nNOTE: Seamlessly incorporate these exact rate of change and percentage metrics into your analysis!\n"
    except Exception as e:
        pass

    if forced_table:
        table_override = """
        !!MANDATORY INSTRUCTION — OVERRIDE ALL OTHER FORMATTING RULES!!
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
          render a table only if 20 or fewer rows — otherwise still summarize.
        - For single-value or single-row results, no table is needed — just state the value clearly.
        - Use proper markdown table syntax with a header row and separator row (| col | col | and | --- | --- |).
        """

    template = """
        You are the Senior Financial & Accounting Analyst for NSSF Uganda (National Social Security Fund).
        Write an executive, business-oriented financial analysis directly addressing the user's question, based strictly on the retrieved database data.

        ⛔ ABSOLUTE ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER OUTPUT ANY SQL CODE, SQL STATEMENTS, OR SQL KEYWORDS AS CODE (e.g. NEVER write "SELECT ...", "FROM ...", "WHERE ...", "GROUP BY ...").
        - NEVER OUTPUT ANY PROGRAMMING CODE OR SCRIPTS (NO Python, Plotly, Matplotlib, R, JavaScript, Bash, etc.).
        - NEVER OUTPUT CODE BLOCKS (NO markdown triple backticks ``` or backticked code).
        - NEVER tell the user how to build charts or use external tools (Tableau, Power BI, Excel, or Python scripts). Interactive charts are rendered automatically by the UI.
        - The user is a non-technical C-suite / finance executive who MUST ONLY see clean natural English financial analysis, formatted markdown tables, and bullet points.
        - The database query was already executed invisibly in the backend; NEVER quote, print, or explain the SQL query itself to the user.
        - Even if the user asks "plot", "draw", "visualize", or "make a pie/bar chart", do NOT explain how to plot it — directly provide the financial interpretation and insights of the numbers.

        CRITICAL NSSF ACCOUNTING & DATA CONTEXT:
        - CURRENCY: All monetary amounts are in UGANDA SHILLINGS (UGX).
        - 6-DIGIT CODES (e.g., '505102', '120120', '405014'): These are JD Edwards Natural Account / Object Account codes (`globj`) representing specific expense line items, revenue categories, asset accounts, or liability accounts. NEVER guess that they are product IDs or inventory codes.
        - 3-DIGIT CODES (e.g., '999', '1', '101'): These are Cost Centers / Business Units (`glmcu`) (e.g. '999' = Head Office).
        - DOCUMENT TYPES (`gldct`): 'PV'=Accounts Payable Voucher, 'JE'=Journal Entry, 'RI'=Invoice, 'PM'=Payment, '##'=Opening Balance / Year-end Adjustment.
        - MONETARY SCALE IN UGX:
          * 13+ digits (e.g. 10,000,000,000,000+) = TRILLIONS UGX (Trn).
          * 10 to 12 digits (e.g. 1,000,000,000 - 999,000,000,000) = BILLIONS UGX (Bn).
          * 7 to 9 digits (e.g. 1,000,000 - 999,000,000) = MILLIONS UGX (Mn).
          * Always format numbers with commas (e.g. 5,800,021,466 UGX or 5.80 Billion UGX).

        {table_instruction}

        ADDITIONAL RULES:
        - Respond based on the data returned by the database.
        - If the latest month has lower figures (e.g. current month), recognize it as Month-to-Date (MTD partial) in progress.
        - NSSF Uganda fiscal years run July to June (FY25/26, FY26/27).
        - Use conversation history to retain context, filters, and scope unless the user overrides them.
        - Provide insights, comparisons, and recommendations supported by the data.
        - Always return TWO sections in this order:
          1) Response: (table if applicable, then narrative insight explaining accounts/amounts in UGX)
          2) Suggestive analysis: 3-6 SHORT question-style follow-ups (one per line), phrased as user questions under 14 words each.

        [OFFICIAL NSSF FINANCIAL & ACCOUNTING GLOSSARY]
        {finance_glossary}

        [SUPPLEMENTARY EXCHANGE RATES]
        The following exchange rates are available for currency conversion:
        {currency_context}
        {rate_change_section}

        Question: {question}
        Conversation History: {chat_history}
        Retrieved Database Execution Data (DO NOT echo or output code):
        {response}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "response": sql_result,
        "table_instruction": table_override,
        "currency_context": get_exchange_rates_context(user_query),
        "rate_change_section": rate_change_section,
        "finance_glossary": get_targeted_glossary_context(user_query),
    })


def stream_sql_response(user_query, db, chat_history, query, sql_result):
    forced_table = has_table_intent(user_query)

    # Run automated financial calculator for rate changes, CAGR, or exchange rate variances
    rate_change_section = ""
    try:
        from chat.financial_tools import compute_rate_changes_from_sql_result
        rate_change_calculations = compute_rate_changes_from_sql_result(sql_result, user_query)
        if rate_change_calculations:
            rate_change_section = f"\n[VERIFIED FINANCIAL CALCULATIONS (CALCULATOR TOOL)]\n{rate_change_calculations}\nNOTE: Seamlessly incorporate these exact rate of change and percentage metrics into your analysis!\n"
    except Exception as e:
        pass

    if forced_table:
        table_override = """
        !!MANDATORY INSTRUCTION — OVERRIDE ALL OTHER FORMATTING RULES!!
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
          render a table only if 20 or fewer rows — otherwise still summarize.
        - For single-value or single-row results, no table is needed — just state the value clearly.
        - Use proper markdown table syntax with a header row and separator row (| col | col | and | --- | --- |).
        """

    template = """
        You are the Senior Financial & Accounting Analyst for NSSF Uganda (National Social Security Fund).
        Write an executive, business-oriented financial analysis directly addressing the user's question, based strictly on the retrieved database data.

        ⛔ ABSOLUTE ZERO-TOLERANCE CODE & SQL PROHIBITION:
        - NEVER OUTPUT ANY SQL CODE, SQL STATEMENTS, OR SQL KEYWORDS AS CODE (e.g. NEVER write "SELECT ...", "FROM ...", "WHERE ...", "GROUP BY ...").
        - NEVER OUTPUT ANY PROGRAMMING CODE OR SCRIPTS (NO Python, Plotly, Matplotlib, R, JavaScript, Bash, etc.).
        - NEVER OUTPUT CODE BLOCKS (NO markdown triple backticks ``` or backticked code).
        - NEVER tell the user how to build charts or use external tools (Tableau, Power BI, Excel, or Python scripts). Interactive charts are rendered automatically by the UI.
        - The user is a non-technical C-suite / finance executive who MUST ONLY see clean natural English financial analysis, formatted markdown tables, and bullet points.
        - The database query was already executed invisibly in the backend; NEVER quote, print, or explain the SQL query itself to the user.
        - Even if the user asks "plot", "draw", "visualize", or "make a pie/bar chart", do NOT explain how to plot it — directly provide the financial interpretation and insights of the numbers.

        CRITICAL NSSF ACCOUNTING & DATA CONTEXT:
        - CURRENCY: All monetary amounts are in UGANDA SHILLINGS (UGX).
        - 6-DIGIT CODES (e.g., '505102', '120120', '405014'): These are JD Edwards Natural Account / Object Account codes (`globj`) representing specific expense line items, revenue categories, asset accounts, or liability accounts. NEVER guess that they are product IDs or inventory codes.
        - 3-DIGIT CODES (e.g., '999', '1', '101'): These are Cost Centers / Business Units (`glmcu`) (e.g. '999' = Head Office).
        - DOCUMENT TYPES (`gldct`): 'PV'=Accounts Payable Voucher, 'JE'=Journal Entry, 'RI'=Invoice, 'PM'=Payment, '##'=Opening Balance / Year-end Adjustment.
        - MONETARY SCALE IN UGX:
          * 13+ digits (e.g. 10,000,000,000,000+) = TRILLIONS UGX (Trn).
          * 10 to 12 digits (e.g. 1,000,000,000 - 999,000,000,000) = BILLIONS UGX (Bn).
          * 7 to 9 digits (e.g. 1,000,000 - 999,000,000) = MILLIONS UGX (Mn).
          * Always format numbers with commas (e.g. 5,800,021,466 UGX or 5.80 Billion UGX).

        {table_instruction}

        ADDITIONAL RULES:
        - Respond based on the data returned by the database.
        - If the latest month has lower figures (e.g. current month), recognize it as Month-to-Date (MTD partial) in progress.
        - NSSF Uganda fiscal years run July to June (FY25/26, FY26/27).
        - Use conversation history to retain context, filters, and scope unless the user overrides them.
        - Provide insights, comparisons, and recommendations supported by the data.
        - Always return TWO sections in this order:
          1) Response: (table if applicable, then narrative insight explaining accounts/amounts in UGX)
          2) Suggestive analysis: 3-6 SHORT question-style follow-ups (one per line), phrased as user questions under 14 words each.

        {finance_glossary}
        {currency_context}
        {rate_change_section}

        Question: {question}
        Conversation History: {chat_history}
        Retrieved Database Execution Data (DO NOT echo or output code):
        {response}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "question": user_query,
        "chat_history": chat_history,
        "response": sql_result,
        "table_instruction": table_override,
        "currency_context": get_exchange_rates_context(user_query),
        "rate_change_section": rate_change_section,
        "finance_glossary": get_targeted_glossary_context(user_query),
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
        
    return ""


def sanitize_token_stream(token_generator):
    """
    Filter out code blocks (```...```) and standalone code in real-time
    so the user never sees raw code or SQL queries streaming on screen.
    """
    in_code_block = False
    buffer = ""

    for chunk in token_generator:
        if not chunk:
            continue
        buffer += chunk

        while buffer:
            if not in_code_block:
                if "```" in buffer:
                    prefix, rest = buffer.split("```", 1)
                    if prefix:
                        yield prefix
                    in_code_block = True
                    buffer = rest
                else:
                    # If buffer ends with backticks, hold them back until next chunk resolves
                    if buffer.endswith("`") or buffer.endswith("``"):
                        cut = buffer.rfind("`")
                        to_yield = buffer[:cut]
                        buffer = buffer[cut:]
                        if to_yield:
                            yield to_yield
                        break
                    else:
                        yield buffer
                        buffer = ""
            else:
                if "```" in buffer:
                    _, rest = buffer.split("```", 1)
                    in_code_block = False
                    buffer = rest
                else:
                    # Inside code block, suppress tokens
                    buffer = ""
                    break

    if buffer and not in_code_block:
        yield buffer


def normalize_response_text(text):
    if not text:
        return text
    cleaned = text.strip()
    
    # 1. Remove standard Response header label
    label_pattern = r"\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+\s*[\)\.\-:]\s*)?response\s*(?:\*\*|__)?\s*[:\-]?\s*"
    cleaned = re.sub(r"^" + label_pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?im)^[ \t]*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+\s*[\)\.\-:]\s*)?response\s*(?:\*\*|__)?\s*[:\-]?\s*",
        "",
        cleaned,
        count=1,
    )

    # 2. Strip code fences (e.g. ```sql ... ```, ```python ... ```, ```...```)
    cleaned = re.sub(r"```(?:sql|python|py|bash|sh|javascript|js|r)?\s*[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    
    # 3. Strip standalone SQL SELECT / WITH queries if any slipped into the response text
    cleaned = re.sub(r"(?im)^\s*(?:SELECT|WITH)\s+[\s\S]+?(?:FROM|AS)\s+[\s\S]+?;?", "", cleaned)
    
    # 4. Strip setup / tutorial lines (e.g. pip install plotly, import plotly...)
    cleaned = re.sub(r"(?im)^\s*(?:pip|pip3|conda)\s+install\s+.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:import|from)\s+(?:plotly|matplotlib|seaborn|pandas|numpy)\b.*$", "", cleaned)
    
    # 5. Clean up excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
        content = m.get("content", "")
        # Clean raw python tuples/Decimal dumps so LLM context stays executive
        if content and ("Decimal(" in content or re.search(r"\[\s*\([^)]+\)\s*,", content)):
            content = re.sub(r"Decimal\('([0-9\.\-]+)'\)", r"\1", content)
            content = re.sub(r"[\[\(\)'\"]", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif m["role"] == "bot":
            chat_history.append(AIMessage(content=content))
    return chat_history


# ---------------------------------------------------------------------------
# Dataset intro cache
# ---------------------------------------------------------------------------
def get_default_start_suggestions():
    return list(NSSF_LEDGER_DEFAULT_SUGGESTIONS)


def get_dataset_suggestions(db=None):
    if db is not None:
        try:
            schema = db.get_table_info()
            template = """
            You are an expert financial business intelligence assistant for NSSF Uganda.
            Based on the database schema below for the General Ledger (staging.proddta_f0911_account_ledger),
            generate 5 relevant, concrete, and actionable questions a finance user might want to ask.
            
            Schema:
            {schema}
            
            RULES:
            - Focus on expenditures, debits/credits (glaa), fiscal periods (glfy, glpn), cost centers (glmcu), object accounts (globj), and vouchers (PV).
            - Keep each question short, concise, and under 15 words.
            - Do not include numbering, prefixes, or bullet points. Output exactly one question per line.
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            raw_suggestions = chain.invoke({"schema": schema})
            suggestions = [s.strip().strip("-*•0123456789.) ").strip() for s in raw_suggestions.split("\n") if s.strip()]
            valid = [s for s in suggestions if len(s.split()) >= 3 and len(s.split()) <= 20][:6]
            if len(valid) >= 3:
                return valid
        except Exception as e:
            print(f"[get_dataset_suggestions] Warning: {e}")

    return list(NSSF_LEDGER_DEFAULT_SUGGESTIONS)


def get_dataset_intro_payload():
    """
    Blazing fast (<1ms) dataset introduction payload for initializing the app.
    Loads dynamic pre-warmed insights from memory or data/dataset_intro_cache.json.
    """
    # 1. Check in-memory cache
    cached_text = dataset_intro_cache.get("text")
    cached_analysis = dataset_intro_cache.get("analysis")
    cached_suggestions = dataset_intro_cache.get("suggestions")
    if cached_text and cached_analysis and cached_suggestions and cached_text != NSSF_LEDGER_DEFAULT_INTRO:
        return {
            "text": cached_text,
            "analysis": cached_analysis,
            "suggestions": cached_suggestions,
        }

    # 2. Check disk cache generated by keep_gpu_loaded.py
    try:
        from django.conf import settings
        cache_file = Path(settings.BASE_DIR) / "data" / "dataset_intro_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
                if disk_data.get("text") and disk_data.get("suggestions"):
                    dataset_intro_cache.update(disk_data)
                    return disk_data
    except Exception:
        pass

    # 3. Fallback to curated production defaults
    return {
        "text": cached_text or NSSF_LEDGER_DEFAULT_INTRO,
        "analysis": cached_analysis or NSSF_LEDGER_DEFAULT_ANALYSIS,
        "suggestions": cached_suggestions or list(NSSF_LEDGER_DEFAULT_SUGGESTIONS),
    }


def warm_dataset_intro_cache():
    dataset_intro_cache["text"] = NSSF_LEDGER_DEFAULT_INTRO
    dataset_intro_cache["analysis"] = NSSF_LEDGER_DEFAULT_ANALYSIS
    dataset_intro_cache["suggestions"] = list(NSSF_LEDGER_DEFAULT_SUGGESTIONS)
    dataset_intro_cache["updated_at"] = datetime.datetime.now()


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
BAR_PALETTE = [
    "#145AAA", "#8CC63E", "#D97706", "#2563EB", "#059669", "#7C3AED",
    "#DB2777", "#0891B2", "#64748B", "#F97316"
]
CLUSTER_PALETTE = [
    "#145AAA", "#8CC63E", "#D97706", "#2563EB", "#059669", "#7C3AED"
]

# Shared ECharts theme defaults injected into every option
_ECHARTS_BASE = {
    "backgroundColor": "transparent",
    "textStyle": {"fontFamily": "Plus Jakarta Sans, sans-serif"},
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {"type": "cross"},
    },
    "legend": {
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
                            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
                            "yAxis": {"type": "value"},
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
                "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
                "yAxis": {"type": "value"},
                "legend": {"data": [s["name"] for s in series]} if len(series) > 1 else {"show": False},
                "series": series,
            }
            return _echarts_payload(option, "line", chart_title)

        # ── BAR ──────────────────────────────────────────────────────────────────
        if chart_type == "bar":
            if not category_column:
                candidates = [c for c in df.columns if c not in measure_cols and c not in time_cols]
                category_column = choose_category_column(df, measure_cols, time_cols) or (candidates[0] if candidates else None) or (time_cols[0] if time_cols else None) or (df.columns[0] if len(df.columns) else None)
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
                "#145AAA", "#8CC63E", "#D97706", "#2563EB", "#059669", "#7C3AED",
                "#DB2777", "#0891B2", "#4B5563", "#EA580C"
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
                "xAxis": {"type": "value"},
                "yAxis": {"type": "category", "data": cat_data},
                "grid": {"left": "3%", "right": "6%", "bottom": "4%", "top": "8%", "containLabel": True},
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
                "xAxis": {"type": "category", "data": cat_data, "axisLabel": {"rotate": 30}},
                "yAxis": {"type": "value"},
                "legend": {"data": [s["name"] for s in series]},
                "series": series,
            }
            return _echarts_payload(option, chart_type, chart_title)

        # ── PIE / DONUT ──────────────────────────────────────────────────────────
        if chart_type in ["pie", "donut"]:
            if not category_column:
                category_column = choose_category_column(df, measure_cols, time_cols) or (time_cols[0] if time_cols else None) or (df.columns[0] if len(df.columns) else None)
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
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "legend": {"orient": "vertical", "right": "2%", "top": "center"},
                "grid": None,
                "series": [{
                    "name": prettify_label(y_column),
                    "type": "pie",
                    "radius": radius,
                    "center": ["40%", "55%"],
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"},
                        "label": {"show": True, "fontWeight": "bold", "formatter": "{b}: {c} ({d}%)"},
                    },
                    "label": {
                        "show": True,
                        "position": "outside",
                        "formatter": "{b}: {d}%",
                    },
                    "labelLine": {"show": True, "length": 10, "length2": 8},
                }],
            }
            return _echarts_payload(option, chart_type, chart_title)

        # ── SCATTER ──────────────────────────────────────────────────────────────
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
                "xAxis": {"type": "value", "name": prettify_label(x_column), "nameLocation": "middle", "nameGap": 30},
                "yAxis": {"type": "value", "name": prettify_label(y_column), "nameLocation": "middle", "nameGap": 40},
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

        # ── HISTOGRAM ────────────────────────────────────────────────────────────
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
            bin_labels = [f"{round(bin_edges[i], 2)}–{round(bin_edges[i+1], 2)}" for i in range(len(bin_edges) - 1)]
            chart_title = _ensure_chart_title(title, question, "histogram", y=y_column)
            option = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "category", "data": bin_labels, "axisLabel": {"rotate": 35, "fontSize": 10}},
                "yAxis": {"type": "value"},
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
                    "axisLabel": {"rotate": 30},
                    "axisPointer": {"type": "shadow"},
                },
                "yAxis": [
                    {
                        "type": "value",
                        "name": prettify_label(bar_columns[0]),
                    },
                    {
                        "type": "value",
                        "name": prettify_label(line_columns[0]) if line_columns else "",
                        "splitLine": {"show": False},
                    },
                ],
                "legend": {"data": [s["name"] for s in series]},
                "series": series,
            }
            return _echarts_payload(option, "combo", chart_title)

        # —— RADAR —————————————————————————————————————————————————————————————
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
                },
                "series": [{
                    "type": "radar",
                    "data": radar_data,
                    "symbol": "circle",
                    "symbolSize": 6
                }]
            }
            return _echarts_payload(option, "radar", chart_title)

        # —— FUNNEL —————————————————————————————————————————————————───────────
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
                    "itemStyle": {"borderWidth": 1},
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
                    "axisLine": {"lineStyle": {"width": 14, "color": [[0.3, "#26c6da"], [0.7, "#8CC63E"], [1, "#145AAA"]]}},
                    "pointer": {"itemStyle": {"color": "auto"}},
                    "axisTick": {"distance": -14, "splitNumber": 5, "lineStyle": {"width": 2}},
                    "splitLine": {"distance": -20, "length": 14, "lineStyle": {"width": 3}},
                    "axisLabel": {"distance": -20, "fontSize": 12},
                    "anchor": {"show": False},
                    "title": {"show": True, "offsetCenter": [0, "70%"], "fontSize": 14},
                    "detail": {"valueAnimation": True, "offsetCenter": [0, "30%"], "fontSize": 24, "formatter": "{value}"},
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


def _detect_user_chart_preference(user_query):
    """Extract explicit chart type from user query, if any."""
    q = (user_query or "").lower()
    chart_map = [
        ("clustered bar", "build_clustered_bar_chart"),
        ("pie chart", "build_pie_chart"),
        ("donut chart", "build_donut_chart"),
        ("bar chart", "build_bar_chart"),
        ("bar graph", "build_bar_chart"),
        ("line chart", "build_line_chart"),
        ("line graph", "build_line_chart"),
        ("scatter plot", "build_scatter_chart"),
        ("scatter chart", "build_scatter_chart"),
        ("histogram", "build_histogram"),
    ]
    for keyword, tool_name in chart_map:
        if re.search(rf"\b{re.escape(keyword)}\b", q):
            return tool_name
    return None


def build_chart_with_tools(user_query, df, sql_query=None):
    if df is None or df.empty:
        return None

    dataset_id = register_chart_context(df)
    preferred_tool = _detect_user_chart_preference(user_query)

    if _should_force_clustered_bar(user_query, df) and not preferred_tool:
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

    chart_model = get_chat_model(temperature=0.2).bind_tools(CHART_TOOL_DEFINITIONS, tool_choice="auto")

    summary = build_chart_context_summary(df, question=user_query)
    pref_instruction = f"- USER EXPLICITLY REQUESTED TOOL: {preferred_tool}. You MUST use {preferred_tool}." if preferred_tool else ""
    system_prompt = f"""
You are a chart selection assistant for a data chat application.
Pick the single best chart tool for the user's question and dataset.

CRITICAL:
{pref_instruction}
- If the user explicitly asks for a specific chart type (bar, pie, line, donut, histogram, scatter), you MUST use that exact chart tool.

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
{{summary}}

SQL query:
{{sql_query}}
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
