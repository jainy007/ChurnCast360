from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np
import subprocess
import mlflow
import uvicorn
import xgboost as xgb

# Paths
MODEL_DIR = "models"
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# FastAPI app
app = FastAPI(title="ChurnCast360 FastAPI", version="1.0")


# Request schema
class InputData(BaseModel):
    features: list


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500, detail="Model not found. Please train first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load imputer
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)

    # Load feature names
    import json

    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    # Prepare input
    input_array = np.array(data.features).reshape(1, -1)

    if input_array.shape[1] != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {input_array.shape[1]}.",
        )

    # Impute if needed
    input_array = imputer.transform(input_array)

    # Predict
    dmatrix = xgb.DMatrix(input_array)
    proba = model.predict(dmatrix)
    prediction = int(proba[0] >= 0.5)

    return {"prediction": prediction, "probability": float(proba[0])}


# Train endpoint
@app.post("/train")
def train():
    try:
        # Trigger training pipeline (reuse restore_pipeline.py or direct pipeline command)
        result = subprocess.run(
            ["python", "src/model/train_tuned_models.py", "--device", "auto"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"status": "training completed", "logs": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")


# Optional: Endpoint to view MLflow URI
@app.get("/mlflow")
def get_mlflow_uri():
    return {"mlflow_tracking_uri": mlflow.get_tracking_uri()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
