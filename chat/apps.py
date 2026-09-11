import os
import time
import subprocess
import threading
import requests
from django.apps import AppConfig


def _start_ollama_if_needed(host, model, keep_alive):
    """
    Check if Ollama is reachable. If not, launch `ollama serve` and wait
    for it to become ready, then preload the model into VRAM.
    """
    def is_alive():
        try:
            r = requests.get(f"{host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    if not is_alive():
        print("[Ollama] Not running — starting ollama serve ...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except FileNotFoundError:
            print("[Ollama] WARNING: 'ollama' executable not found on PATH.")
            return

        # Wait up to 15 s for Ollama to become ready
        for _ in range(15):
            time.sleep(1)
            if is_alive():
                break
        else:
            print("[Ollama] WARNING: Ollama did not become ready in 15 s.")
            return

    # Set multi-model environment variables so Ollama keeps both in VRAM
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "3")
    os.environ.setdefault("OLLAMA_KEEP_ALIVE", "24h")

    # Use langchain_ollama ChatOllama to preload — same config as utils.py.
    # request_timeout=600 gives larger models enough time to load into VRAM.
    # We stream and break on first chunk to confirm the model is resident in VRAM.
    try:
        from langchain_ollama import ChatOllama
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        loader = ChatOllama(
            model=model,
            base_url=host,
            keep_alive=keep_alive,
            num_ctx=num_ctx,
            temperature=0,
            request_timeout=600,
        )
        for _ in loader.stream("hi"):
            break
        print(f"[Ollama] {model} loaded into GPU VRAM — will persist for {keep_alive}.")
    except Exception as e:
        print(f"[Ollama] Preload warning for {model}: {e}")


def _refresh_db_intro():
    """
    Query live PostgreSQL for real stats, then ask the LLM to write
    a natural description and suggestive questions from those facts.
    Runs in a background thread so server startup is non-blocking.
    """
    try:
        import datetime
        from sqlalchemy import text
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from chat.utils import init_database, dataset_intro_cache, llm

        db = init_database()
        engine = db._engine

        stats_sql = """
            SELECT
                COUNT(*)                                                   AS total_rows,
                MIN(glfy)                                                  AS min_fy,
                MAX(glfy)                                                  AS max_fy,
                COUNT(DISTINCT TRIM(glmcu))                                AS cost_centers,
                COUNT(DISTINCT TRIM(globj))                                AS accounts,
                COUNT(DISTINCT gldct)                                      AS doc_types,
                COUNT(DISTINCT glfy)                                       AS fiscal_years,
                SUM(CASE WHEN glaa > 0 THEN glaa ELSE 0 END)              AS total_debits,
                SUM(CASE WHEN glaa < 0 THEN ABS(glaa) ELSE 0 END)        AS total_credits,
                TO_DATE((1900000 + MAX(gldgj))::text, 'YYYYDDD')         AS latest_date,
                TO_DATE((1900000 + MIN(gldgj))::text, 'YYYYDDD')         AS earliest_date
            FROM staging.proddta_f0911_account_ledger
            WHERE glpost = 'P'
        """

        with engine.connect() as conn:
            row = conn.execute(text(stats_sql)).fetchone()

        if row is None:
            return

        total_rows    = f"{row[0]:,}"
        min_fy        = 1900 + row[1] if row[1] else "N/A"
        max_fy        = 1900 + row[2] if row[2] else "N/A"
        cost_centers  = row[3]
        accounts      = row[4]
        doc_types     = row[5]
        fiscal_years  = row[6]
        debits_bn     = row[7] / 1_000_000_000 if row[7] else 0
        credits_bn    = row[8] / 1_000_000_000 if row[8] else 0
        latest_date   = row[9].strftime("%d %b %Y") if row[9] else "N/A"
        earliest_date = row[10].strftime("%d %b %Y") if row[10] else "N/A"

        facts = f"""
Database table: staging.proddta_f0911_account_ledger (NSSF Uganda General Ledger)
Total posted transactions: {total_rows}
Fiscal years covered: {fiscal_years} years, from FY{min_fy} to FY{max_fy}
Earliest transaction date: {earliest_date}
Latest transaction date: {latest_date}
Distinct cost centres (glmcu): {cost_centers}
Distinct natural accounts (globj): {accounts}
Distinct document types (gldct): {doc_types}
Total debits (positive glaa): UGX {debits_bn:,.2f} Billion
Total credits (negative glaa): UGX {credits_bn:,.2f} Billion
Net balance: UGX {(debits_bn - credits_bn):,.2f} Billion
""".strip()

        template = """You are writing the welcome description for a financial BI chatbot
connected to the NSSF Uganda General Ledger database.

Below are the REAL, LIVE statistics queried directly from the database right now:

{facts}

Task 1 — Write a concise, professional 2-3 sentence description of what this dataset
contains and covers. Use the actual numbers above. Do NOT invent figures.
Start with: "This chatbot is connected to..."

Task 2 — Write exactly 6 short, specific, actionable questions a finance analyst
would want to ask about this data. Use real fiscal years and metrics from the stats above.
One question per line. No numbering, bullets, or prefixes.

Format your response as:
DESCRIPTION:
<your 2-3 sentences here>

SUGGESTIONS:
<question 1>
<question 2>
<question 3>
<question 4>
<question 5>
<question 6>
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"facts": facts})

        # Parse the LLM response
        intro, suggestions = "", []
        if "DESCRIPTION:" in raw and "SUGGESTIONS:" in raw:
            desc_part = raw.split("SUGGESTIONS:")[0].replace("DESCRIPTION:", "").strip()
            sugg_part = raw.split("SUGGESTIONS:")[1].strip()
            intro = desc_part
            suggestions = [
                s.strip().strip("-*•0123456789.) ").strip()
                for s in sugg_part.split("\n") if s.strip()
            ]
            suggestions = [s for s in suggestions if len(s.split()) >= 3][:6]

        if not intro:
            intro = raw.strip()

        analysis = (
            "Run trend analysis, period comparisons (MoM / YoY / QoQ), drill-downs by "
            "cost centre (glmcu), natural account (globj), subledger (glsbl), or document "
            "type (gldct). Monitor expenditure vs budget, generate interactive charts, "
            "and export data — all queried live from the warehouse."
        )

        dataset_intro_cache.update({
            "text": intro,
            "analysis": analysis,
            "suggestions": suggestions or list(dataset_intro_cache.get("suggestions", [])),
            "updated_at": datetime.datetime.now(),
        })
        print(f"[AppReady] LLM-generated intro ready: {total_rows} rows, FY{min_fy}–{max_fy}.")

    except Exception as e:
        print(f"[AppReady] DB intro refresh skipped (will use defaults): {e}")


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        # Only run once — StatReloader spawns a child process with RUN_MAIN=true
        if os.environ.get("RUN_MAIN") != "true":
            return

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        sql_model = os.getenv("OLLAMA_SQL_MODEL", "pxlksr/defog_sqlcoder-7b-2:Q8")
        nlp_model = os.getenv("OLLAMA_NLP_MODEL", "qwen2.5:7b")
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "24h")

        # Preload both models into Ollama GPU memory (non-blocking)
        threading.Thread(
            target=_start_ollama_if_needed,
            args=(host, sql_model, keep_alive),
            daemon=True,
        ).start()
        threading.Thread(
            target=_start_ollama_if_needed,
            args=(host, nlp_model, keep_alive),
            daemon=True,
        ).start()

        # Fetch real DB stats in a background thread — updates intro cache when ready
        threading.Thread(target=_refresh_db_intro, daemon=True).start()
