import pandas as pd
import random
from faker import Faker
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
fake = Faker()

def load_telco_churn_kaggle():
    path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["source"] = "kaggle_telco"
    return df

def load_ibm_telco():
    path = RAW_DIR / "IBM-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["source"] = "ibm_telco"
    return df

def load_bank_churn():
    path = RAW_DIR / "bank_churn.csv"
    df = pd.read_csv(path)
    df.rename(columns={"Exited": "churn"}, inplace=True)
    df["source"] = "bank_churn"
    return df

def generate_saas_churn(n=2000):
    data = []
    for _ in range(n):
        data.append({
            "customer_id": fake.uuid4(),
            "signup_date": fake.date_between(start_date='-2y', end_date='-1y'),
            "last_active": fake.date_between(start_date='-1y', end_date='today'),
            "country": fake.country(),
            "subscription_tier": random.choice(["Free", "Basic", "Pro", "Enterprise"]),
            "monthly_spend": round(random.uniform(0, 200), 2),
            "support_tickets": random.randint(0, 10),
            "churn": random.choices([0, 1], weights=[0.8, 0.2])[0],
            "source": "synthetic_saas"
        })
    return pd.DataFrame(data)

def harmonize_columns(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).fillna(df["churn"]).astype(int)
    return df

def save_processed(df, name):
    path = PROCESSED_DIR / f"{name}_cleaned.csv"
    df.to_csv(path, index=False)
    print(f"Saved cleaned data: {path}")

def run():
    dfs = {
        "kaggle_telco": load_telco_churn_kaggle(),
        "ibm_telco": load_ibm_telco(),
        "bank_churn": load_bank_churn(),
        "synthetic_saas": generate_saas_churn(),
    }

    for name, df in dfs.items():
        df_cleaned = harmonize_columns(df)
        save_processed(df_cleaned, name)

if __name__ == "__main__":
    run()
