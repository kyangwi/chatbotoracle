import os
import re
import urllib.parse
import pandas as pd
import oracledb
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Oracle 11.2 requires Thick mode in python-oracledb
try:
    oracledb.init_oracle_client()
except Exception:
    try:
        # Fallback path if auto-detection fails
        instant_client_path = r"C:\instantclient_19\instantclient_21_22"
        oracledb.init_oracle_client(lib_dir=instant_client_path)
    except Exception as e:
        print(f"Warning: Failed to load Oracle Client in Thick Mode: {e}")

def sanitize_name(name: str) -> str:
    # Remove any non-alphanumeric character (keep underscores)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    # Remove leading underscores/digits if they make it invalid for Oracle
    cleaned = re.sub(r'^[^a-zA-Z]+', '', cleaned)
    # Convert to uppercase
    cleaned = cleaned.upper()
    # Limit to 30 characters (Oracle 11g limit)
    cleaned = cleaned[:30]
    # Ensure it's not empty
    if not cleaned:
        cleaned = "COL_NAME"
    return cleaned

def create_schema_if_not_exists():
    admin_user = os.getenv("DB_USER", "SYSTEM")
    admin_password = os.getenv("DB_PASSWORD", "admin123")
    database = os.getenv("DB_NAME", "xe")

    safe_password = urllib.parse.quote_plus(admin_password)
    # Connect as administrator
    admin_uri = f"oracle+oracledb://{admin_user}:{safe_password}@localhost:1521/?service_name={database}"
    engine = create_engine(admin_uri)

    print("Checking if 'ADVENTUREWORK' user exists...")
    with engine.connect() as conn:
        # Oracle usernames are uppercase by default
        result = conn.execute(text("SELECT COUNT(*) FROM dba_users WHERE username = 'ADVENTUREWORK'"))
        exists = result.scalar() > 0

        if not exists:
            print("Creating user/schema 'ADVENTUREWORK'...")
            conn.execute(text("CREATE USER adventurework IDENTIFIED BY adventurework"))
            conn.execute(text("GRANT CONNECT, RESOURCE, DBA TO adventurework"))
            conn.execute(text("ALTER USER adventurework DEFAULT TABLESPACE users QUOTA UNLIMITED ON users"))
            conn.commit()
            print("Successfully created 'ADVENTUREWORK' user and granted privileges.")
        else:
            print("User/schema 'ADVENTUREWORK' already exists.")

def inject_csv_files():
    # Schema credentials we created
    schema_user = "adventurework"
    schema_password = "adventurework"
    database = os.getenv("DB_NAME", "xe")

    schema_uri = f"oracle+oracledb://{schema_user}:{schema_password}@localhost:1521/?service_name={database}"
    engine = create_engine(schema_uri)

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]

    if not csv_files:
        print("\n" + "="*80)
        print("NO CSV FILES FOUND!")
        print("-"*80)
        print(f"Please put your CSV files in the '{data_dir}' folder and run this script again.")
        print("="*80 + "\n")
        return

    print(f"Found {len(csv_files)} CSV file(s) to inject.")

    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        # Table name is the filename (without extension) sanitized
        raw_table_name = os.path.splitext(file_name)[0]
        table_name = sanitize_name(raw_table_name)

        print(f"\nProcessing '{file_name}' -> Table '{table_name}'...")

        try:
            # Read CSV with encoding fallback
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, falling back to Latin1 encoding...")
                df = pd.read_csv(file_path, encoding='latin1')
            
            # Clean and sanitize column names
            original_cols = df.columns
            df.columns = [sanitize_name(col) for col in df.columns]

            # Detect duplicates in sanitized column names and append number if needed
            col_counts = {}
            new_cols = []
            for col in df.columns:
                if col in col_counts:
                    col_counts[col] += 1
                    new_cols.append(f"{col}_{col_counts[col]}")
                else:
                    col_counts[col] = 1
                    new_cols.append(col)
            df.columns = new_cols

            print("Sanitized Columns:")
            for orig, new in zip(original_cols, df.columns):
                if orig != new:
                    print(f"  '{orig}' -> '{new}'")

            # Map float64 columns to standard Numeric to avoid Oracle binary_precision error
            from sqlalchemy.types import Numeric
            dtype_map = {}
            for col in df.columns:
                if pd.api.types.is_float_dtype(df[col]):
                    dtype_map[col] = Numeric()

            # Write to Oracle
            df.to_sql(
                name=table_name.lower(), # Write lowercase name, SQLAlchemy handles quote names
                con=engine,
                if_exists='replace',
                index=False,
                chunksize=1000,
                dtype=dtype_map
            )
            print(f"Successfully loaded {len(df)} rows into table '{table_name}'.")

        except Exception as e:
            print(f"Error loading '{file_name}': {e}")

if __name__ == "__main__":
    try:
        create_schema_if_not_exists()
        inject_csv_files()
    except Exception as e:
        print(f"Initialization/injection error: {e}")
