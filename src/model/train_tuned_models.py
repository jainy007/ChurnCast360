import pandas as pd
import sqlite3
import os
import pickle
import json
import argparse
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

"""
Script to train multiple baseline models (Logistic Regression, Random Forest, XGBoost)
on churn prediction data sourced from SQLite.
Includes preprocessing, label encoding, imputation, and evaluation metrics.
Automatically handles CPU/GPU execution for XGBoost.
"""

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    choices=["auto", "cpu", "cuda"],
    default="auto",
    help="Device for XGBoost",
)
args = parser.parse_args()

# Device selection for XGBoost
xgb_device = "cpu"
if args.device == "cuda":
    xgb_device = "cuda"
elif args.device == "auto":
    try:
        # Test if XGBoost can use CUDA
        test_dmatrix = xgb.DMatrix([[1], [2]], label=[0, 1])
        xgb.train(
            {"tree_method": "hist", "device": "cuda"}, test_dmatrix, num_boost_round=1
        )
        xgb_device = "cuda"
    except Exception:
        xgb_device = "cpu"

print(f"XGBoost version: {xgb.__version__}")
print(f"Using device for XGBoost: {xgb_device}")

# Step 1: Load Data from SQLite
DB_PATH = os.path.join(os.getcwd(), "churncast360.db")
conn = sqlite3.connect(DB_PATH)
print("Connected to database")

df = pd.read_sql("SELECT * FROM churn_features", conn)
conn.close()
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Preprocessing
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

label_encoders_path = os.path.join("models", "label_encoders.pkl")
os.makedirs("models", exist_ok=True)
with open(label_encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)
print("Label encoders saved")

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

imputer_path = os.path.join("models", "imputer.pkl")
with open(imputer_path, "wb") as f:
    pickle.dump(imputer, f)
print("Imputer saved")
print("Features and target prepared")

# Step 3: Train-test split
target_col = "churn"
X = df_imputed.drop(columns=[target_col])
y = df_imputed[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Model Training and Evaluation
metrics_summary = {}


def evaluate_model(name, model, X_test, y_test, y_pred, y_proba):
    with mlflow.start_run(run_name=name):
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        model_path = os.path.join("models", f"{name.lower()}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        # MLflow logging
        mlflow.log_param("model_name", name)
        if name.lower() == "randomforest":
            mlflow.log_param("n_estimators", model.n_estimators)
        elif name.lower() == "logisticregression":
            mlflow.log_param("max_iter", model.max_iter)
        elif name.lower() == "xgboost":
            mlflow.log_param("num_boost_round", 100)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc", auc)

        mlflow.log_artifact(model_path)

        metrics_summary[name] = {
            "accuracy": accuracy,
            "recall": recall,
            "auc": auc,
            "classification_report": report,
        }


# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
evaluate_model(
    "LogisticRegression",
    log_model,
    X_test,
    y_test,
    log_model.predict(X_test),
    log_model.predict_proba(X_test)[:, 1],
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(
    "RandomForest",
    rf_model,
    X_test,
    y_test,
    rf_model.predict(X_test),
    rf_model.predict_proba(X_test)[:, 1],
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
evaluate_model("XGBoost", xgb_model, X_test, y_test, y_pred, y_proba)

# Step 5: Save Metrics Summary
metrics_path = os.path.join("models", "metrics_summary.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_summary, f, indent=4)
print(f"Metrics summary saved to {metrics_path}")
