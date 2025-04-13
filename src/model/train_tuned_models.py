import pandas as pd
import sqlite3
import os
import pickle
import json
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import mlflow

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
print(f"[DEBUG] Using MLFLOW_TRACKING_URI: {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)

# Constants
EXCLUDE_COLUMNS = ["customer_id", "customerid"]
TARGET_COLUMN = "churn"
MODEL_DIR = Path("models")
DB_PATH = os.path.join("data", "churncast360.db")


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    choices=["auto", "cpu", "cuda"],
    default="auto",
    help="Device for XGBoost",
)
args = parser.parse_args()


def determine_xgb_device():
    if args.device == "cuda":
        return "cuda"
    elif args.device == "auto":
        try:
            test_dmatrix = xgb.DMatrix([[1], [2]], label=[0, 1])
            xgb.train(
                {"tree_method": "hist", "device": "cuda"},
                test_dmatrix,
                num_boost_round=1,
            )
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


def load_data():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM churn_features", conn)
    conn.close()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Label Encoding
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [
        col for col in categorical_cols if col not in EXCLUDE_COLUMNS + [TARGET_COLUMN]
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    print("Label encoders saved")

    # Drop unnecessary columns
    df = df.drop(columns=EXCLUDE_COLUMNS, errors="ignore")

    # Separate features and target
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    # Imputation
    imputer = SimpleImputer(strategy="median")
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features), columns=features.columns
    )

    with open(MODEL_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    print("Imputer saved")

    return features_imputed, target


def train_test_split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(list(X_train.columns), f)
    return X_train, X_test, y_train, y_test


def evaluate_and_log_model(
    name, model, X_test, y_test, y_pred, y_proba, metrics_summary
):
    with mlflow.start_run(run_name=name):
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(
            f"\nModel: {name}\nAccuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nAUC-ROC: {auc:.4f}"
        )
        print(classification_report(y_test, y_pred))

        model_path = MODEL_DIR / f"{name.lower()}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        # MLflow logging
        mlflow.log_param("model_name", name)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_iter"):
            mlflow.log_param("max_iter", model.max_iter)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc", auc)
        mlflow.log_artifact(str(model_path))

        metrics_summary[name] = {
            "accuracy": accuracy,
            "recall": recall,
            "auc": auc,
            "classification_report": report,
        }


def train_models(X_train, X_test, y_train, y_test):
    metrics_summary = {}

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    evaluate_and_log_model(
        "LogisticRegression",
        log_model,
        X_test,
        y_test,
        log_model.predict(X_test),
        log_model.predict_proba(X_test)[:, 1],
        metrics_summary,
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_and_log_model(
        "RandomForest",
        rf_model,
        X_test,
        y_test,
        rf_model.predict(X_test),
        rf_model.predict_proba(X_test)[:, 1],
        metrics_summary,
    )

    # XGBoost
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": xgb_device,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    y_proba = xgb_model.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)
    evaluate_and_log_model(
        "XGBoost", xgb_model, X_test, y_test, y_pred, y_proba, metrics_summary
    )

    return metrics_summary


def save_metrics_summary(metrics_summary):
    with open(MODEL_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Metrics summary saved.")


def main():
    global xgb_device
    xgb_device = determine_xgb_device()
    print(f"Determined XGBoost device: {xgb_device}")

    df = load_data()
    features, target = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(features, target)
    metrics_summary = train_models(X_train, X_test, y_train, y_test)
    save_metrics_summary(metrics_summary)


if __name__ == "__main__":
    main()
