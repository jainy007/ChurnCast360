# SETUP

## Install Pre-requisites

First, create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate churncast360
```

## Code Quality
We use Black and Ruff for formatting and linting.

### Auto-fix issues locally:

```
pre-commit run --all-files
```

### Install Azurite (Azure Blob Simulator)
```
sudo npm install -g azurite@3.22.0
```

### Run Azurite in a separate terminal:

```
azurite --location ./azurite --debug ./azurite/debug.log --loose
```

### Install Azure SDK for Python
```
pip install azure-storage-blob
```

## PIPELINE: PROCESS AND LOAD DATASETS

```
python src/ingest/load_datasets.py
```

**What it does:**

- Loads Kaggle Telco, IBM Telco, Bank Churn, and Synthetic Faker data
- Saves raw and cleaned processed files in /data/

## DATABASE INTEGRATION (SQLite)

```
python src/sql/create_unified_view.py
```

**What it does:**

- Connects to churncast360.db
- Creates unified SQL view churn_features
- Prepares data for modeling from SQL

## AZURE BLOB SIMULATOR

Upload datasets to Azurite blob storage:

```
python src/utils/azure_blob_simulator.py
```

**What it does:**

- Uploads raw datasets to the local Azure Blob Simulator
- Downloads test blob to verify connectivity
- Lists available blobs for verification
- Note: Ensure Azurite is running before executing this script.

## BASELINE MODEL TRAINING

**Goals:**

- Load data from SQLite (churn_features view)
- Handle missing values using median imputation
- Encode categorical variables and save encoders
- Train Logistic Regression baseline model
- Evaluate with Accuracy, Recall, and AUC-ROC
- Save trained model and imputer to models/ directory

**Command:**

```
python src/model/train_baseline_model.py
```

## DOCKER & FASTAPI + MLflow ORCHESTRATION

### Step 1: Build Docker images and bring up services

From project root directory:

```
docker compose build --no-cache
docker compose up
```

****What it does:****

- Brings up FastAPI at http://127.0.0.1:8000
- Brings up MLflow UI at http://127.0.0.1:5000
- Docker Compose is configured to:
- Reuse existing models/, data/, and src/ directories inside the container
- Create fresh /mlflow/artifacts and mlflow/mlflow.db inside the container

### Step 2: Test the API Endpoints
Run the test runner to automatically validate FastAPI endpoints:

```
python src/test_app_endpoints.py
```

**This will:**

- Check /health endpoint
- Check /predict endpoint with sample payload
- Prompt if you want to trigger /train endpoint
- If /train is triggered, MLflow will record the experiment

## Minikube Kubernetes Orchestration (Minikube Flow)

## Step 1: Start Minikube

```
minikube start
eval $(minikube docker-env)
```

### Step 2: Build images inside Minikube Docker daemon

```
docker build -t churncast360-mlflow -f docker/mlflow/Dockerfile .
docker build -t churncast360-fastapi -f docker/fastapi/Dockerfile .
```

### Step 3: Deploy MLflow and FastAPI services
```
kubectl apply -f k8s/mlflow/deployment.yaml
kubectl apply -f k8s/mlflow/service.yaml
kubectl apply -f k8s/fastapi/deployment.yaml
kubectl apply -f k8s/fastapi/service.yaml
```

### Step 4: Verify pods and services

```
kubectl get pods
kubectl get svc
```

### Step 5: Test Minikube endpoints
Get the Minikube service URLs:

```
minikube service churncast-fastapi-service --url
minikube service churncast-mlflow-service --url
```

Run test runner with Minikube mode:

```
python src/test_app_endpoints.py --minikube
```

What it does:

- Checks /health endpoint
- Checks /predict endpoint
- Optional /train endpoint, with MLflow logging to Minikube MLflow service
```


## Notes

- /data endpoint is not yet implemented.
- MLflow will show runs only if the /train endpoint is successful.
- models/ folder is kept in Git for reproducibility.
- data/ folder is ignored to avoid size bloat and LFS issues.
- .dockerignore is cleaned for Docker build context efficiency.
- .gitignore properly ignores heavy data directories.