import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = Path("reports/eda/plots")
SUMMARY_DIR = Path("reports/eda/summaries")


def summarize_dataset(df: pd.DataFrame, name: str):
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df["churn"].value_counts(normalize=True).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    with open(SUMMARY_DIR / f"{name}_summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)
    print(f"Saved summary for {name}")


def plot_churn_distribution(df: pd.DataFrame, name: str):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="churn", data=df)
    plt.title(f"Churn Distribution - {name}")
    plt.savefig(PLOTS_DIR / f"{name}_churn_dist.png")
    plt.close()


def plot_top_numeric_correlations(df: pd.DataFrame, name: str):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "churn" not in numeric_cols:
        return

    corr = df[numeric_cols].corr()["churn"].sort_values(key=abs, ascending=False)[1:6]
    corr.plot(kind="barh", title=f"Top Correlations with Churn - {name}")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name}_top_corrs.png")
    plt.close()


def run():
    for file in PROCESSED_DIR.glob("*_cleaned.csv"):
        name = file.stem.replace("_cleaned", "")
        df = pd.read_csv(file)

        summarize_dataset(df, name)
        plot_churn_distribution(df, name)
        plot_top_numeric_correlations(df, name)


if __name__ == "__main__":
    run()
