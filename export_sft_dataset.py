"""
Export SFT Dataset from Glossary _ Finance Chatbot Training.xlsx
Generates high-quality Supervised Fine-Tuning (SFT) training pairs in OpenAI/ShareGPT JSONL format.
"""

import os
import json
import pandas as pd

EXCEL_FILE = "Glossary _ Finance Chatbot Training.xlsx"
OUTPUT_JSONL = "dataset_sft.jsonl"


def generate_sft_dataset():
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: {EXCEL_FILE} not found.")
        return

    xl = pd.ExcelFile(EXCEL_FILE)
    dataset = []

    print(f"Reading {EXCEL_FILE}...")

    # 1. Financial Domain Acronyms & Accounting Rules (Sheets 1 to 7)
    for sheet in xl.sheet_names:
        if sheet == "F0911 Table":
            continue
        df = pd.read_excel(xl, sheet_name=sheet)
        for _, row in df.iterrows():
            term = str(row.iloc[1]).strip()
            meaning = str(row.iloc[2]).strip()
            if not term or term.lower() in ["term / jargon", "nan"]:
                continue

            clean_meaning = " ".join(meaning.split())

            # QA Style 1: Direct Definition
            dataset.append({
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert financial and accounting assistant for NSSF Uganda specializing in {sheet}."
                    },
                    {
                        "role": "user",
                        "content": f"What is '{term}' and how is it defined in NSSF finance operations?"
                    },
                    {
                        "role": "assistant",
                        "content": f"In NSSF Uganda finance operations, **{term}** refers to: {clean_meaning}"
                    }
                ]
            })

            # QA Style 2: Concept Inversion
            dataset.append({
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert financial and accounting assistant for NSSF Uganda specializing in {sheet}."
                    },
                    {
                        "role": "user",
                        "content": f"Which term or acronym represents '{clean_meaning}'?"
                    },
                    {
                        "role": "assistant",
                        "content": f"The term for '{clean_meaning}' is **{term}**."
                    }
                ]
            })

    # 2. ERP F0911 General Ledger Column Dictionary (Sheet 8)
    if "F0911 Table" in xl.sheet_names:
        df_f0911 = pd.read_excel(xl, sheet_name="F0911 Table")
        for _, row in df_f0911.iterrows():
            col = str(row["Field"]).strip().lower()
            desc = str(row["Description"]).strip()
            dtype = str(row.get("Data Type", "")).strip()

            # Column Definition QA
            dataset.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a PostgreSQL Financial Data Engineer for NSSF Uganda General Ledger (staging.proddta_f0911_account_ledger)."
                    },
                    {
                        "role": "user",
                        "content": f"Which column stores '{desc}' in table staging.proddta_f0911_account_ledger?"
                    },
                    {
                        "role": "assistant",
                        "content": f"In `staging.proddta_f0911_account_ledger`, **{desc}** is stored in column `{col}` (Data Type: {dtype})."
                    }
                ]
            })

    # Write out JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Exported {len(dataset)} SFT training pairs to {OUTPUT_JSONL}.")


if __name__ == "__main__":
    generate_sft_dataset()
