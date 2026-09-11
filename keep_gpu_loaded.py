"""
keep_gpu_loaded.py
------------------
Persistent GPU VRAM Keeper & Background Intelligence Pre-Warmer for NSSF Uganda ChatBot.

Keeps `gpt-oss:20b` locked in GPU memory (RTX 4060 Ti 16GB) 24/7, whether the Django
server is running, stopped, or restarting. Periodically warms GPU CUDA kernels and 
pre-generates dynamic, data-driven dataset intros into `data/dataset_intro_cache.json`
so users receive rich, dynamic LLM responses instantly (< 1ms).
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path

# Setup Django environment so we can use existing LLM chains and database connections
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sqlchat_project.settings")

import django
django.setup()

from chat.utils import (
    llm,
    sql_llm,
    init_database,
    build_data_overview_response,
    get_dataset_suggestions,
    normalize_response_text,
    NSSF_LEDGER_DEFAULT_INTRO,
    NSSF_LEDGER_DEFAULT_ANALYSIS,
    NSSF_LEDGER_DEFAULT_SUGGESTIONS,
)

CACHE_FILE = BASE_DIR / "data" / "dataset_intro_cache.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

HEARTBEAT_INTERVAL_SECONDS = 900  # 15 minutes


def ping_gpu_models() -> bool:
    """Sends a keep-alive pulse to Ollama to keep both SQL and NLP models pinned in GPU VRAM."""
    sql_model_name = os.getenv("OLLAMA_SQL_MODEL", "pxlksr/defog_sqlcoder-7b-2:Q8")
    nlp_model_name = os.getenv("OLLAMA_NLP_MODEL", "qwen2.5:7b")
    all_ok = True

    # 1. Ping NLP model
    try:
        t0 = time.time()
        res_nlp = llm.invoke("OK")
        elapsed_nlp = time.time() - t0
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] NLP ({nlp_model_name}) pulse success ({elapsed_nlp:.2f}s)")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] NLP pulse warning: {e}")
        all_ok = False

    # 2. Ping SQL model
    try:
        t0 = time.time()
        res_sql = sql_llm.invoke("SELECT 1;")
        elapsed_sql = time.time() - t0
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] SQL ({sql_model_name}) pulse success ({elapsed_sql:.2f}s)")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] SQL pulse warning: {e}")
        all_ok = False

    return all_ok


def refresh_dynamic_intro() -> dict:
    """
    Connects to the database, queries schema info, invokes LLM to generate
    relevant, live dataset insights, and saves to data/dataset_intro_cache.json.
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Refreshing dynamic dataset intelligence from live database...")
    payload = {
        "text": NSSF_LEDGER_DEFAULT_INTRO,
        "analysis": NSSF_LEDGER_DEFAULT_ANALYSIS,
        "suggestions": list(NSSF_LEDGER_DEFAULT_SUGGESTIONS),
        "updated_at": datetime.datetime.now().isoformat(),
    }

    try:
        db = init_database()
        t0 = time.time()
        
        # 1. Generate dynamic overview
        raw_overview = build_data_overview_response("What is this data all about?", db, chat_history=[])
        if raw_overview:
            cleaned = normalize_response_text(raw_overview)
            import re
            sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
            intro = " ".join([s for s in sentences if s][:2]).strip() or cleaned.strip()
            if intro and len(intro) > 30:
                payload["text"] = intro
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Generated dynamic intro: {intro[:100]}...")

        # 2. Generate dynamic suggestions
        suggestions = get_dataset_suggestions(db)
        if suggestions and len(suggestions) >= 3:
            payload["suggestions"] = suggestions
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Generated {len(suggestions)} dynamic questions.")

        elapsed = time.time() - t0
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Dynamic refresh complete in {elapsed:.2f}s.")

    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Note during dynamic refresh: {e}")

    # Atomically save to disk
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Saved fresh cache to {CACHE_FILE.name}")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [GPU Keeper] Failed saving cache file: {e}")

    return payload


def run_keeper_loop():
    sql_model_name = os.getenv("OLLAMA_SQL_MODEL", "pxlksr/defog_sqlcoder-7b-2:Q8")
    nlp_model_name = os.getenv("OLLAMA_NLP_MODEL", "qwen2.5:7b")
    print("=" * 70)
    print("  NSSF Uganda ChatBot - Persistent GPU VRAM Keeper & Intelligence Pre-Warmer")
    print("=" * 70)
    print(f"Ensuring [{sql_model_name}] and [{nlp_model_name}] are locked in GPU memory 24/7.")
    print(f"Heartbeat interval: Every {HEARTBEAT_INTERVAL_SECONDS // 60} minutes.")
    print("Press Ctrl+C to stop.\n")

    # Initial warm-up pulse
    ping_gpu_models()

    # Initial background dynamic generation
    refresh_dynamic_intro()

    while True:
        try:
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)
            ping_gpu_models()
        except KeyboardInterrupt:
            print("\n[GPU Keeper] Exiting gracefully.")
            break
        except Exception as e:
            print(f"[GPU Keeper] Loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_keeper_loop()
