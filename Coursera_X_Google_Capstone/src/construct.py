# construct.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df = pd.get_dummies(df, columns=["department", "salary"], drop_first=True)

    X = df.drop("left", axis=1)
    y = df["left"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def check_multicollinearity(X_train):

    vif_data = pd.DataFrame()
    vif_data["feature"] = [f"X{i}" for i in range(X_train.shape[1])]
    vif_data["VIF"] = [
        variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])
    ]
    print("\nVIF Scores:\n", vif_data)


def build_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    DATA_DIR = Path("data")
    X_train, X_test, y_train, y_test = load_and_prepare(
        f"{DATA_DIR}/clean_sailfort_employee_data.csv"
    )
    check_multicollinearity(X_train)
    model = build_model(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
