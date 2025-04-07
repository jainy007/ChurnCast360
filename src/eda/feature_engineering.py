import pandas as pd
import time
from pathlib import Path
from pandarallel import pandarallel

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

# Start timer
start_time = time.time()

PROCESSED_DIR = Path("data/processed")
FINAL_PATH = Path("data/processed/master_dataset.csv")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop high-null columns or fill where possible
    df = df.dropna(axis=1, thresh=len(df) * 0.7)  # drop cols with >30% missing
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Convert boolean-style objects
    for col in df.select_dtypes(include="object"):
        if df[col].nunique() <= 5:
            df[col] = df[col].astype("category")

    # One-hot encode categorical vars
    df = pd.get_dummies(df, drop_first=True)

    # Clip outliers using pandarallel
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].parallel_apply(
        lambda col: col.clip(lower=col.quantile(0.01), upper=col.quantile(0.99))
    )

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 6, 12, 24, 60, 100],
            labels=["<6mo", "6-12", "12-24", "24-60", "60+"],
        )
        df = pd.get_dummies(df, columns=["tenure_bucket"], drop_first=True)

    if {"monthly_spend", "support_tickets"}.issubset(df.columns):
        df["spend_per_ticket"] = df["monthly_spend"] / (df["support_tickets"] + 1)

    return df


def run():
    dfs = []

    for file in PROCESSED_DIR.glob("*_cleaned.csv"):
        df = pd.read_csv(file)
        df = preprocess(df)
        df = add_engineered_features(df)
        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    combined.to_csv(FINAL_PATH, index=False)
    print(f"Saved master dataset with shape: {combined.shape}")

    # End timer
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
