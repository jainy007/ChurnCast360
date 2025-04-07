import os
import json
import pytest

MODEL_DIR = 'models'

@pytest.mark.parametrize("model_file", [
    "logisticregression_model.pkl",
    "randomforest_model.pkl",
    "xgboost_model.pkl"
])
def test_model_files_exist(model_file):
    model_path = os.path.join(MODEL_DIR, model_file)
    assert os.path.exists(model_path), f"Model file {model_file} is missing!"

def test_metrics_summary_exists():
    metrics_path = os.path.join(MODEL_DIR, 'metrics_summary.json')
    assert os.path.exists(metrics_path), "metrics_summary.json is missing!"

def test_metrics_summary_content():
    metrics_path = os.path.join(MODEL_DIR, 'metrics_summary.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    assert metrics, "Metrics summary is empty!"

    for model_name, model_metrics in metrics.items():
        for metric in ['accuracy', 'recall', 'auc']:
            value = model_metrics.get(metric)
            assert value is not None, f"{metric} missing for {model_name}"
            assert 0.0 <= value <= 1.0, f"{metric} out of range for {model_name}"

# Optional: future enhancement for MLflow experiment check
