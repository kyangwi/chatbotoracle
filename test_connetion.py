import oracledb
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv,find_dotenv
import urllib.parse
import os 

load_dotenv(find_dotenv())

# Oracle 11.2 requires Thick mode in python-oracledb
try:
    oracledb.init_oracle_client()
except Exception:
    try:
        # If you downloaded 64-bit Oracle Instant Client, extract it and update the path below:
        instant_client_path = r"C:\instantclient_19\instantclient_21_22"
        oracledb.init_oracle_client(lib_dir=instant_client_path)
    except Exception as e:
        print("\n" + "="*80)
        print("ORACLE CLIENT ERROR (ARCHITECTURE MISMATCH / MISSING LIBRARIES)")
        print("-"*80)
        print("Your Python is 64-bit, but your local Oracle Database XE is 32-bit.")
        print("A 64-bit Python process cannot load 32-bit DLLs (oci.dll) from your server bin directory.")
        print("\nTO FIX THIS:")
        print("1. Download 'Instant Client Basic Lite' (ZIP file) for Windows x64 (64-bit) from:")
        print("   https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html")
        print("2. Extract the ZIP file (e.g., to C:\\oracle\\instantclient_19).")
        print("3. Update the 'instant_client_path' variable in 'test_connetion.py' to your extracted folder path.")
        print("="*80 + "\n")
        raise RuntimeError("Failed to load Oracle 64-bit client libraries required for Oracle 11.2.")

def init_database(user, password, database) -> SQLDatabase:
    if not user or not password:
        raise ValueError(
            "Missing database credentials! Please create a '.env' file in the root directory "
            "with the following content:\n\n"
            "DB_USER=adventurework\n"
            "DB_PASSWORD=adventurework\n"
            "DB_NAME=xe\n"
            "GEMINI_API_KEY=your_google_api_key\n"
        )
    
    # URL-encode the password to handle any special characters safely
    safe_password = urllib.parse.quote_plus(password)
    # Using oracle+oracledb connection string for local Oracle instance (xe)
    db_uri = f"oracle+oracledb://{user}:{safe_password}@localhost:1521/?service_name={database}"
    return SQLDatabase.from_uri(db_uri)

def get_schema_text():
    db = init_database(
        os.getenv("DB_USER", "adventurework"),
        os.getenv("DB_PASSWORD", "adventurework"),
        os.getenv("DB_NAME", "xe"),
    )
    schema = db.get_table_info()
    return schema

# Example usage
print(get_schema_text())

