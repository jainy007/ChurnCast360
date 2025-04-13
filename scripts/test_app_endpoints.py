import pandas as pd
import pickle
import json
from pathlib import Path
import subprocess
import argparse

MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/master_dataset.csv")

# Set default URL, override via argument
FASTAPI_URL = "http://127.0.0.1:8000"


def run_curl(command):
    print(f"\nRunning command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print("Response:\n", result.stdout.strip())
    if result.stderr:
        print("Errors:\n", result.stderr.strip())
    print("-" * 50)


def check_health():
    command = ["curl", f"{FASTAPI_URL}/health"]
    run_curl(command)


def generate_sample_payload():
    # Load label encoders
    with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    # Load imputer
    with open(MODEL_DIR / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    # Load feature names
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    # Load raw data (first row only)
    df_sample = pd.read_csv(DATA_PATH, nrows=1)

    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df_sample.columns:
            df_sample[col] = le.transform(df_sample[col].astype(str))

    # Align columns
    for col in feature_names:
        if col not in df_sample.columns:
            df_sample[col] = 0

    df_sample = df_sample[feature_names]

    # Impute
    df_imputed = pd.DataFrame(imputer.transform(df_sample), columns=df_sample.columns)

    # Prepare payload
    sample = df_imputed.iloc[0].tolist()
    payload = {"features": sample}

    return payload


def test_predict():
    import requests

    payload = generate_sample_payload()

    print("Step 2: Test predict endpoint")
    try:
        response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
        print("Response:")
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")


def trigger_train():
    import requests

    print("Step 3: Trigger training endpoint")
    try:
        response = requests.post(f"{FASTAPI_URL}/train")
        print("Response:")
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FastAPI endpoints")
    parser.add_argument(
        "--minikube",
        action="store_true",
        help="Use Minikube endpoint instead of localhost",
    )
    args = parser.parse_args()

    if args.minikube:
        # Get minikube IP dynamically
        result = subprocess.run(["minikube", "ip"], capture_output=True, text=True)
        minikube_ip = result.stdout.strip()
        FASTAPI_URL = f"http://{minikube_ip}:30080"
        print(f"Using Minikube endpoint: {FASTAPI_URL}")
    else:
        print(f"Using Docker Compose endpoint: {FASTAPI_URL}")

    print("Step 1: Check health endpoint")
    check_health()

    test_predict()

    user_input = (
        input("Step 3: Do you want to trigger /train endpoint? (y/n): ").strip().lower()
    )
    if user_input == "y":
        trigger_train()
    else:
        print("Skipped training trigger.")
