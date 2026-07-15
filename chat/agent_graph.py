"""
LangGraph-based agentic pipeline for the SQL ChatBot.

Replaces the linear get_response_with_query() chain with a fault-tolerant
state machine that classifies intent, generates/executes/self-heals SQL,
and produces structured responses — all with Pydantic-validated state.
"""

import json
import logging
import traceback
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .utils import (
    llm,
    sql_llm,
    gemini_model,
    normalize_response_text,
    extract_clean_sql,
    build_data_overview_response,
    build_no_sql_response,
    build_sql_response,
    get_exchange_rates_context,
    is_data_overview_question,
    is_sql_query_question,
    has_visual_intent,
    should_generate_chart,
    build_chart_with_tools,
    fetch_dataframe,
    get_sql_chain,
)

logger = logging.getLogger("agent_graph")


# ---------------------------------------------------------------------------
# Pydantic Models — typed state at every boundary
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    """Possible intent classifications for a user message."""
    SQL_QUERY = "sql_query"
    NO_SQL = "no_sql"
    DATA_OVERVIEW = "data_overview"


class IntentClassification(BaseModel):
    """Structured output from the intent classifier."""
    intent: IntentType = Field(
        description="The classified intent of the user's message"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the classification"
    )


class SQLResult(BaseModel):
    """Wraps the outcome of a SQL execution attempt."""
    success: bool = Field(default=False)
    data: Optional[str] = Field(default=None, description="Raw result from DB")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    query: str = Field(default="", description="The SQL query that was executed")


class AgentResponse(BaseModel):
    """The final structured output from the agent graph."""
    response_text: str = Field(default="")
    sql_query: Optional[str] = Field(default=None)
    sql_result: Optional[str] = Field(default=None)
    chart_payload: Optional[dict] = Field(default=None)
    intent: Optional[str] = Field(default=None)
    retries_used: int = Field(default=0)
    error: Optional[str] = Field(default=None)


class AgentGraphState(TypedDict, total=False):
    """
    TypedDict state schema for the LangGraph StateGraph.

    Using TypedDict (not Pydantic BaseModel) is required by LangGraph
    to properly preserve all keys across node transitions. Each node
    returns a partial dict and LangGraph merges it into this schema.
    """
    # --- Inputs ---
    user_query: str
    chat_history: list
    db: Any  # SQLDatabase instance — preserved as a live object reference

    # --- Intent ---
    intent: Optional[str]
    intent_reasoning: str

    # --- SQL pipeline ---
    sql_query: Optional[str]
    sql_result: Optional[str]
    sql_error: Optional[str]
    sql_retries: int

    # --- Response ---
    response_text: str
    chart_payload: Optional[dict]

    # --- Error tracking ---
    node_errors: list

    # --- Final output ---
    final_response: Optional[dict]


# ---------------------------------------------------------------------------
# Maximum retries for self-healing SQL
# ---------------------------------------------------------------------------
MAX_SQL_RETRIES = 2


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def classify_intent(state: dict) -> dict:
    """
    Node 1: Classify the user's intent using structured heuristics first,
    then fall back to LLM if ambiguous. Returns the intent enum value.
    """
    try:
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])

        # Fast-path: use the existing regex heuristics first (zero LLM cost)
        if is_data_overview_question(user_query):
            return {
                "intent": IntentType.DATA_OVERVIEW,
                "intent_reasoning": "Matched data overview heuristic patterns",
            }

        if is_sql_query_question(user_query):
            return {
                "intent": IntentType.SQL_QUERY,
                "intent_reasoning": "Matched database-centric heuristic patterns (revenue/sales/trend/monthly/average/KPIs)",
            }

        # Use LLM for ambiguous cases — ask it to classify with structured output
        classification_prompt = ChatPromptTemplate.from_template("""
You are an intent classifier for a business intelligence SQL chatbot.
Classify the user's message into exactly one of these intents:

- "sql_query": The user wants data from the database (counts, trends, comparisons, 
  specific records, charts/graphs of data, KPIs, etc.)
- "no_sql": The user is making small talk, greetings, asking general questions 
  not about the database, or asking about capabilities
- "data_overview": The user wants a high-level summary of what the entire dataset 
  contains (e.g., "what is this data about?", "describe the dataset")

Recent conversation for context:
{chat_history}

User message: {question}

Respond with ONLY a JSON object: {{"intent": "sql_query"|"no_sql"|"data_overview", "reasoning": "brief explanation"}}
""")

        chain = classification_prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "question": user_query,
            "chat_history": str(chat_history[-6:]) if chat_history else "[]",
        })

        # Parse the LLM's JSON response with Pydantic validation
        cleaned = raw.strip()
        # Extract JSON from potential markdown code blocks
        if "```" in cleaned:
            import re
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)

        parsed = IntentClassification.model_validate_json(cleaned)
        return {
            "intent": parsed.intent,
            "intent_reasoning": parsed.reasoning,
        }

    except Exception as e:
        logger.warning(f"Intent classification failed, defaulting to sql_query: {e}")
        # Safe default: treat as SQL query (the SQL chain will return NO_SQL if needed)
        return {
            "intent": IntentType.SQL_QUERY,
            "intent_reasoning": f"Classification failed ({e}), defaulting to sql_query",
            "node_errors": state.get("node_errors", []) + [f"classify_intent: {e}"],
        }


def generate_sql(state: dict) -> dict:
    """
    Node 2 (SQL branch): Generate SQL from the user's question.
    Uses the existing get_sql_chain but wraps output with validation.
    """
    try:
        db = state.get("db")
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])

        sql_chain = get_sql_chain(db)
        raw_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        }).strip()

        raw_query = extract_clean_sql(raw_query)

        if raw_query == "NO_SQL":
            # The SQL chain decided this doesn't need SQL — reclassify
            return {
                "intent": IntentType.NO_SQL,
                "sql_query": None,
                "intent_reasoning": "SQL chain returned NO_SQL",
            }

        return {"sql_query": raw_query}

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {
            "sql_query": None,
            "sql_error": f"SQL generation failed: {e}",
            "node_errors": state.get("node_errors", []) + [f"generate_sql: {e}"],
        }


def execute_sql(state: dict) -> dict:
    """
    Node 3: Execute the generated SQL against the database.
    Captures errors structurally for the self-healing retry loop.
    """
    try:
        db = state.get("db")
        sql_query = state.get("sql_query")

        if not sql_query:
            return {
                "sql_error": "No SQL query to execute",
                "sql_result": None,
            }

        result = db.run(sql_query)

        # Validate we got something back
        sql_result_obj = SQLResult(
            success=True,
            data=str(result),
            query=sql_query,
        )

        return {
            "sql_result": sql_result_obj.data,
            "sql_error": None,
        }

    except Exception as e:
        retries = state.get("sql_retries", 0)
        error_msg = str(e)
        logger.warning(f"SQL execution failed (attempt {retries + 1}): {error_msg}")

        return {
            "sql_error": error_msg,
            "sql_result": None,
            "sql_retries": retries + 1,
        }


def fix_sql(state: dict) -> dict:
    """
    Self-healing node: Takes the failed SQL + error message, asks the LLM
    to fix the query, and feeds it back into execute_sql.
    """
    try:
        db = state.get("db")
        original_query = state.get("sql_query", "")
        error_msg = state.get("sql_error", "")
        user_query = state.get("user_query", "")

        fix_prompt = ChatPromptTemplate.from_template("""
You are an expert Oracle Database engineer. A SQL query failed with an error.
Fix the query so it executes successfully.

Original user question: {question}

Database schema: {schema}

Failed SQL query:
{failed_query}

Error message:
{error}

RULES:
- Output ONLY the corrected SQL query, nothing else
- Use Oracle Database syntax (compatible with Oracle 11.2) exclusively. Remember we are using Oracle DB, NOT SQL Server. Do not use SQL Server specific syntax, functions, or operators.
- Do NOT wrap in markdown code blocks
- Do NOT end the query with a semicolon (;). Oracle driver via SQLAlchemy will throw ORA-00911 for trailing semicolons.
- Use uppercase for all table and column names since Oracle stores them in UPPERCASE.
- If the error is about a missing column, check the schema and use the correct column name
- If the error is about syntax (such as using TOP, LIMIT, GETDATE(), etc.), fix it using Oracle equivalents (ROWNUM, SYSDATE, etc.)
- If the error is about CLOB datatypes (e.g. ORA-00932: inconsistent datatypes: expected - got CLOB), it means you are using a CLOB column in a GROUP BY, ORDER BY, DISTINCT, JOIN, or comparison directly. Wrap the CLOB column in TO_CHAR(column_name) or CAST(column_name AS VARCHAR2(4000)) to convert it to a varchar2.
  Example: Use GROUP BY TO_CHAR(country) instead of GROUP BY country.
- Keep the query as close to the original intent as possible
""")

        chain = fix_prompt | sql_llm | StrOutputParser()
        fixed_query = chain.invoke({
            "question": user_query,
            "schema": db.get_table_info(),
            "failed_query": original_query,
            "error": error_msg,
        }).strip()

        fixed_query = extract_clean_sql(fixed_query)

        logger.info(f"Self-healed SQL (attempt {state.get('sql_retries', 0)}): {fixed_query}")

        return {
            "sql_query": fixed_query,
            "sql_error": None,
        }

    except Exception as e:
        logger.error(f"SQL fix attempt failed: {e}")
        return {
            "sql_error": f"Fix attempt also failed: {e}",
            "node_errors": state.get("node_errors", []) + [f"fix_sql: {e}"],
        }


def handle_sql_error(state: dict) -> dict:
    """
    Terminal error handler: when SQL fails after max retries,
    generate a helpful response explaining the issue.
    """
    error_msg = state.get("sql_error", "Unknown error")
    user_query = state.get("user_query", "")

    fallback_text = (
        f"I attempted to query the database for your question but encountered "
        f"a persistent error after {MAX_SQL_RETRIES} attempts. "
        f"The issue was: {error_msg}\n\n"
        f"**Suggestive analysis:**\n"
        f"- Could you rephrase your question with more specific details?\n"
        f"- What specific columns or tables are you interested in?\n"
        f"- Would you like to see the database schema first?\n"
    )

    return {
        "response_text": fallback_text,
        "sql_query": state.get("sql_query"),
    }


def generate_response(state: dict) -> dict:
    """
    Node 4 (SQL success path): Build the natural language response
    from the SQL result using the existing build_sql_response chain.
    """
    try:
        db = state.get("db")
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])
        sql_query = state.get("sql_query", "")
        sql_result = state.get("sql_result", "")

        response_text = build_sql_response(
            user_query, db, chat_history, sql_query, sql_result
        )
        response_text = normalize_response_text(response_text)

        return {"response_text": response_text}

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        # Graceful fallback: return the raw SQL result
        sql_result = state.get("sql_result", "")
        return {
            "response_text": f"Here are the query results:\n\n{sql_result}\n\n"
                            f"*(Response formatting encountered an issue: {e})*",
            "node_errors": state.get("node_errors", []) + [f"generate_response: {e}"],
        }


def handle_data_overview(state: dict) -> dict:
    """
    Node (data_overview branch): Generate a dataset overview response.
    """
    try:
        db = state.get("db")
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])

        response_text = build_data_overview_response(user_query, db, chat_history)
        response_text = normalize_response_text(response_text)

        return {"response_text": response_text}

    except Exception as e:
        logger.error(f"Data overview failed: {e}")
        return {
            "response_text": "This chatbot helps you query and analyze your database "
                            "with natural language. Try asking about specific metrics, "
                            "trends, or comparisons in your data.",
            "node_errors": state.get("node_errors", []) + [f"handle_data_overview: {e}"],
        }


def handle_no_sql(state: dict) -> dict:
    """
    Node (no_sql branch): Handle non-database questions (chitchat, capabilities).
    """
    try:
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history", [])

        response_text = build_no_sql_response(user_query, chat_history)
        response_text = normalize_response_text(response_text)

        return {"response_text": response_text}

    except Exception as e:
        logger.error(f"No-SQL response failed: {e}")
        return {
            "response_text": "I'm a business intelligence chatbot — I can help you "
                            "query, analyze, and visualize your data. "
                            "Try asking a question about your database!",
            "node_errors": state.get("node_errors", []) + [f"handle_no_sql: {e}"],
        }


def maybe_generate_chart(state: dict) -> dict:
    """
    Node 5: Conditionally generate a chart payload if the data supports it.
    Only runs on the SQL success path.
    """
    try:
        db = state.get("db")
        user_query = state.get("user_query", "")
        sql_query = state.get("sql_query")
        chat_history = state.get("chat_history", [])

        if not sql_query or not db:
            return {"chart_payload": None}

        # Check if user explicitly wants a visual OR the data is chart-worthy
        visual_forced = has_visual_intent(user_query)
        df = fetch_dataframe(db, sql_query)

        if df is not None and not df.empty:
            if visual_forced or should_generate_chart(user_query, df):
                chart_payload = build_chart_with_tools(user_query, df, sql_query)
                return {"chart_payload": chart_payload}

        return {"chart_payload": None}

    except Exception as e:
        logger.warning(f"Chart generation failed (non-fatal): {e}")
        return {
            "chart_payload": None,
            "node_errors": state.get("node_errors", []) + [f"maybe_generate_chart: {e}"],
        }


def finalize(state: dict) -> dict:
    """
    Terminal node: Assemble the final AgentResponse.
    This is the single exit point for all branches.
    """
    # Safely extract intent value (could be IntentType enum or raw string)
    raw_intent = state.get("intent")
    if raw_intent is not None:
        intent_str = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
    else:
        intent_str = None

    response = AgentResponse(
        response_text=state.get("response_text", ""),
        sql_query=state.get("sql_query"),
        sql_result=state.get("sql_result"),
        chart_payload=state.get("chart_payload"),
        intent=intent_str,
        retries_used=state.get("sql_retries", 0),
        error="; ".join(state.get("node_errors", [])) if state.get("node_errors") else None,
    )
    # Store the validated Pydantic model as a serialized dict
    return {"final_response": response.model_dump()}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_by_intent(state: dict) -> str:
    """Route after classify_intent based on the detected intent."""
    intent = state.get("intent")
    if intent == IntentType.DATA_OVERVIEW:
        return "handle_data_overview"
    elif intent == IntentType.NO_SQL:
        return "handle_no_sql"
    else:
        return "generate_sql"


def route_after_sql_gen(state: dict) -> str:
    """Route after generate_sql — if it reclassified as NO_SQL, go there."""
    intent = state.get("intent")
    sql_query = state.get("sql_query")

    if intent == IntentType.NO_SQL or not sql_query:
        return "handle_no_sql"
    return "execute_sql"


def route_after_execution(state: dict) -> str:
    """
    Route after execute_sql:
    - Success → generate_response
    - Error + retries left → fix_sql (self-healing)
    - Error + max retries → handle_sql_error (graceful failure)
    """
    sql_error = state.get("sql_error")
    retries = state.get("sql_retries", 0)

    if not sql_error:
        return "generate_response"
    elif retries < MAX_SQL_RETRIES:
        return "fix_sql"
    else:
        return "handle_sql_error"


# ---------------------------------------------------------------------------
# Build and compile the graph (once, at module level)
# ---------------------------------------------------------------------------

def _build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph with all nodes and edges.

    Flow:
        classify_intent ──→ generate_sql ──→ execute_sql ──→ generate_response ──→ maybe_generate_chart ──→ finalize
                       │                          │ ↑                                                          ↑
                       │                   error  │ │ fix_sql                                                  │
                       │                          ↓ │                                                          │
                       │                    handle_sql_error ──────────────────────────────────────────────────→│
                       ├──→ handle_data_overview ──────────────────────────────────────────────────────────────→│
                       └──→ handle_no_sql ─────────────────────────────────────────────────────────────────────→│
    """
    graph = StateGraph(AgentGraphState)

    # Add all nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("fix_sql", fix_sql)
    graph.add_node("handle_sql_error", handle_sql_error)
    graph.add_node("generate_response", generate_response)
    graph.add_node("handle_data_overview", handle_data_overview)
    graph.add_node("handle_no_sql", handle_no_sql)
    graph.add_node("maybe_generate_chart", maybe_generate_chart)
    graph.add_node("finalize", finalize)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional: intent routing
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_data_overview": "handle_data_overview",
            "handle_no_sql": "handle_no_sql",
            "generate_sql": "generate_sql",
        },
    )

    # Conditional: after SQL generation (might reclassify as NO_SQL)
    graph.add_conditional_edges(
        "generate_sql",
        route_after_sql_gen,
        {
            "handle_no_sql": "handle_no_sql",
            "execute_sql": "execute_sql",
        },
    )

    # Conditional: after SQL execution (success / retry / fail)
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "generate_response": "generate_response",
            "fix_sql": "fix_sql",
            "handle_sql_error": "handle_sql_error",
        },
    )

    # fix_sql always loops back to execute_sql
    graph.add_edge("fix_sql", "execute_sql")

    # SQL error handler goes straight to finalize
    graph.add_edge("handle_sql_error", "finalize")

    # Success paths → chart check → finalize
    graph.add_edge("generate_response", "maybe_generate_chart")
    graph.add_edge("maybe_generate_chart", "finalize")

    # Non-SQL branches → finalize
    graph.add_edge("handle_data_overview", "finalize")
    graph.add_edge("handle_no_sql", "finalize")

    # Finalize → END
    graph.add_edge("finalize", END)

    return graph


# Compile the graph once at import time
_compiled_graph = _build_agent_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent_graph(user_query: str, db, chat_history: list) -> AgentResponse:
    """
    Run the full agentic pipeline through the LangGraph state machine.

    Returns an AgentResponse Pydantic model with all structured outputs.

    This is the single entry point that replaces get_response_with_query().
    """
    try:
        # Pass a plain dict matching the AgentGraphState TypedDict schema.
        # The db object is a live SQLDatabase reference — TypedDict state
        # preserves it across all node transitions without serialisation.
        result = _compiled_graph.invoke({
            "user_query": user_query,
            "chat_history": chat_history,
            "db": db,
            "sql_retries": 0,
            "node_errors": [],
        })

        # Extract the finalized response
        final = result.get("final_response")
        if final:
            return AgentResponse.model_validate(final)

        # Fallback: build response from raw state
        raw_intent = result.get("intent")
        intent_val = raw_intent.value if hasattr(raw_intent, "value") else (str(raw_intent) if raw_intent else None)
        return AgentResponse(
            response_text=result.get("response_text", ""),
            sql_query=result.get("sql_query"),
            sql_result=result.get("sql_result"),
            chart_payload=result.get("chart_payload"),
            intent=intent_val,
            retries_used=result.get("sql_retries", 0),
        )

    except Exception as e:
        logger.error(f"Agent graph execution failed: {traceback.format_exc()}")
        # Ultimate fallback: return a structured error response
        return AgentResponse(
            response_text=(
                "I encountered an unexpected error processing your request. "
                "Please try rephrasing your question or ask something else.\n\n"
                "**Suggestive analysis:**\n"
                "- What tables are in this database?\n"
                "- Show me a summary of the data\n"
                "- What are the top KPIs?\n"
            ),
            error=str(e),
        )
