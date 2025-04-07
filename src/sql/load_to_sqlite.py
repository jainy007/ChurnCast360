import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("churncast360.db")
PROCESSED_DIR = Path("data/processed")


def load_cleaned_datasets_to_sqlite():
    conn = sqlite3.connect(DB_PATH)

    for file in PROCESSED_DIR.glob("*_cleaned.csv"):
        df = pd.read_csv(file)
        table_name = file.stem.replace("_cleaned", "")
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded table: {table_name} ({df.shape[0]} rows)")

    conn.commit()
    conn.close()


def preview_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in DB:")
    for t in tables:
        print(f"  • {t[0]}")
    conn.close()


if __name__ == "__main__":
    load_cleaned_datasets_to_sqlite()
    preview_tables()
