# analyze.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(filepath):
    return pd.read_csv(filepath)


def plot_distributions(df):
    sns.histplot(df["satisfaction_level"], kde=True)
    plt.title("Distribution of Satisfaction Level")
    plt.savefig("images/eda_satisfaction_level.png")

    sns.histplot(df["average_monthly_hours"], kde=True)
    plt.title("Distribution of Average Monthly Hours")
    plt.savefig("images/eda_avg_monthly_hrs.png")


def plot_correlation(df):
    numeric_df = df.select_dtypes(include=["number"])  # <== FIX
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix (Numerical Features Only)")
    plt.savefig("images/corr_matrix.png")


def plot_categorical(df):
    sns.countplot(x="salary", hue="left", data=df)
    plt.title("Employee Left by Salary Level")
    plt.savefig("images/emp_left_by_salary_level.png")

    sns.countplot(x="department", hue="left", data=df)
    plt.title("Employee Left by Department")
    plt.xticks(rotation=45)
    plt.savefig("images/emp_left_by_dept.png")


def insights(df):
    print("\nHigh risk segments based on initial EDA:")
    high_risk = df[df["left"] == 1]
    print(high_risk.groupby("salary").size())
    print(high_risk.groupby("department").size())


if __name__ == "__main__":
    DATA_DIR = Path("data")
    df = load_data(f"{DATA_DIR}/clean_sailfort_employee_data.csv")
    plot_distributions(df)
    plot_correlation(df)
    plot_categorical(df)
    insights(df)
