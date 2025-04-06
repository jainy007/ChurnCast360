import pandas as pd
import sqlite3
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

# --- Step 1: Load Data from SQLite --- #
DB_PATH = os.path.join(os.getcwd(), 'churncast360.db')
conn = sqlite3.connect(DB_PATH)

print("Connected to database")

query = "SELECT * FROM churn_features"
df = pd.read_sql(query, conn)
conn.close()

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Step 2: Prepare Data --- #
# Drop any potential ID columns, or non-numeric columns that slipped through
target_col = 'churn'  # ensure your target column is named 'churn'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

X = df.drop(columns=[target_col])
y = df[target_col]

print("Features and target prepared")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")
