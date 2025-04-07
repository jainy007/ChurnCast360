import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("churncast360.db")


def get_table_columns(conn, table_name):
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]


def generate_union_sql(conn, tables, view_name):
    # Collect all columns across all tables
    all_columns = set()
    table_columns = {}
    for table in tables:
        cols = get_table_columns(conn, table)
        table_columns[table] = cols
        all_columns.update(cols)

    # Create a harmonized SELECT for each table
    union_selects = []
    for table in tables:
        cols = []
        for col in sorted(all_columns):
            if col in table_columns[table]:
                cols.append(f"{col}")
            else:
                cols.append(f"NULL AS {col}")
        cols.append(f"'{table}' AS source")
        union_selects.append(f"SELECT {', '.join(cols)} FROM {table}")

    # Combine all into a full UNION ALL view
    return f"CREATE VIEW {view_name} AS\n" + "\nUNION ALL\n".join(union_selects) + ";"


def create_unified_view():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    view_name = "churn_features"
    cursor.execute(f"DROP VIEW IF EXISTS {view_name};")

    # Get list of all churn tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [
        t[0]
        for t in cursor.fetchall()
        if t[0] in {"kaggle_telco", "ibm_telco", "bank_churn", "synthetic_saas"}
    ]

    # Build and execute auto-generated view SQL
    union_sql = generate_union_sql(conn, tables, view_name)
    cursor.execute(union_sql)
    print(f"Created view: {view_name}")

    # Preview rows
    df = pd.read_sql(f"SELECT * FROM {view_name} LIMIT 5", conn)
    print("\n📄 Sample rows:")
    print(df)

    conn.close()


if __name__ == "__main__":
    create_unified_view()
