# plan.py

import pandas as pd
from pathlib import Path


def ingest_data(filepath):
    df = pd.read_csv(filepath)
    return df


def standardize_columns(df):
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    df.rename(
        columns={
            "average_montly_hours": "average_monthly_hours",
            "number_project": "num_projects",
            "time_spend_company": "years_at_company",
        },
        inplace=True,
    )
    return df


def check_missing(df):
    print("\nMissing values per column:\n", df.isnull().sum())


def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")


def check_outliers(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        print(f"{col}: {outliers.shape[0]} outliers")


def initial_eda(df):
    print("\nInitial Data Overview:\n", df.describe())
    print("\nClass distribution:\n", df["left"].value_counts())


def save_clean_data(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    DATA_DIR = Path("data")
    print("DATA DIR = ", DATA_DIR)
    df = ingest_data(f"{DATA_DIR}/HR_comma_sep.csv")
    df = standardize_columns(df)
    check_missing(df)
    check_duplicates(df)
    check_outliers(df)
    initial_eda(df)
    save_clean_data(df, f"{DATA_DIR}/clean_sailfort_employee_data.csv")
