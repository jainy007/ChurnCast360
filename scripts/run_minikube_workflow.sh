#!/bin/bash

set -e
set -o pipefail

RESTORE_PIPELINE=false

# Color functions
function success() { echo -e "\033[32m$1\033[0m"; }
function error() { echo -e "\033[31m$1\033[0m"; }
function warning() { echo -e "\033[33m$1\033[0m"; }

# Pre-check: Make sure Docker Compose is not running
if docker compose ps --format '{{.Names}}' | grep -q .; then
    warning "Docker Compose services are running. Stopping them to avoid Minikube poisoning."
    docker compose down
    success "Docker Compose services stopped."
fi

echo "Running Minikube cleanup."
minikube delete || true

echo "Starting Minikube."
minikube start

# Set Docker to use Minikube daemon
eval $(minikube docker-env)

# Parse flags
if [[ "$1" == "--restore" ]]; then
    RESTORE_PIPELINE=true
fi

if [ "$RESTORE_PIPELINE" = true ]; then
    echo "Restoring pipeline."
    python scripts/restore_pipeline.py || {
        error "Pipeline restore failed."
        minikube stop
        exit 1
    }
else
    echo "Skipping pipeline restore (use --restore to enable)."
fi

echo "Building Docker images inside Minikube."
docker build -t churncast360-mlflow -f docker/mlflow/Dockerfile .
docker build -t churncast360-fastapi -f docker/fastapi/Dockerfile .

echo "Deploying to Kubernetes."
kubectl apply -f k8s/mlflow/deployment.yaml
kubectl apply -f k8s/mlflow/service.yaml
kubectl apply -f k8s/fastapi/deployment.yaml
kubectl apply -f k8s/fastapi/service.yaml

echo "Waiting for Kubernetes pods to become ready."
kubectl wait --for=condition=ready pod -l app=churncast-mlflow --timeout=120s
kubectl wait --for=condition=ready pod -l app=churncast-fastapi --timeout=120s

echo "Testing FastAPI endpoints."
python scripts/test_app_endpoints.py --minikube || {
    error "FastAPI tests failed."
    minikube stop
    exit 1
}

echo "Stopping Minikube."
minikube stop

success "Minikube workflow completed successfully."
