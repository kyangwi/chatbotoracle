import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from chat.utils import init_database

def test_connection():
    print("Testing connection to NSSF Uganda General Ledger Database...")
    db = init_database()
    print("Connected tables:", db.get_usable_table_names())
    print("\nSample Schema Info:")
    print(db.get_table_info())

if __name__ == "__main__":
    test_connection()
