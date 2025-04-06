"""
ChurnCast360 - Baseline Logistic Regression Model Trainer

This script performs the following tasks:
1. Connects to the local SQLite database ('churncast360.db') and loads the unified 'churn_features' view.
2. Preprocesses the data:
   - Encodes all categorical features using Label Encoding.
   - Saves label encoders as 'label_encoders.pkl' for future inference and inverse transformations.
   - Drops non-relevant identifier columns like 'customer_id' (if present).
3. Splits the data into training and testing sets (80/20 split).
4. Trains a Logistic Regression model as the baseline classifier.
5. Evaluates the model using Accuracy, Recall, AUC-ROC, and provides a classification report.
6. Saves the trained model to the 'models/' directory as 'logistic_regression_model.pkl'.

Outputs:
- Trained model: models/logistic_regression_model.pkl
- Saved label encoders: models/label_encoders.pkl
- Evaluation metrics printed to console.

Note:
This script is part of the ChurnCast360 pipeline simulating an enterprise-level churn prediction workflow, preparing for further enhancements like MLflow tracking, automated CI/CD, and containerized deployments.
"""


import pandas as pd
import sqlite3
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# --- Step 0: Paths and Helpers --- #
DB_PATH = os.path.join(os.getcwd(), 'churncast360.db')
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
IMPUTER_PATH = os.path.join(MODEL_DIR, 'imputer.pkl')

def encode_categoricals(df: pd.DataFrame):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
    return df, label_encoders

# --- Step 1: Load Data from SQLite --- #
conn = sqlite3.connect(DB_PATH)
print("Connected to database")

query = "SELECT * FROM churn_features"
df = pd.read_sql(query, conn)
conn.close()

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Step 2: Prepare Data --- #
target_col = 'churn'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Encode categoricals
df, label_encoders = encode_categoricals(df)

# Optional: Save label encoders for future use
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"Label encoders saved to {ENCODER_PATH}")

# Drop identifier columns if any
df = df.drop(columns=["customer_id"], errors="ignore")

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]
# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("Missing values handled with median imputation")


with open(IMPUTER_PATH, 'wb') as f:
    pickle.dump(imputer, f)
print(f"Imputer saved to {IMPUTER_PATH}")

print("Features and target prepared")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# --- Step 3: Train Model --- #
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Logistic Regression model trained")

# --- Step 4: Evaluate --- #
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Step 5: Save Model --- #
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")
