#!/bin/bash

set -e
set -o pipefail

RESTORE_PIPELINE=false

# Color functions
function success() { echo -e "\033[32m$1\033[0m"; }
function error() { echo -e "\033[31m$1\033[0m"; }
function warning() { echo -e "\033[33m$1\033[0m"; }

# Pre-check: Make sure Minikube is not running
if minikube status &>/dev/null; then
    warning "Minikube is running. Stopping Minikube to avoid Docker daemon poisoning."
    minikube stop
    success "Minikube stopped."
fi

# Ensure Docker daemon is active
if ! systemctl is-active --quiet docker; then
    warning "Docker daemon is not running. Starting Docker."
    sudo systemctl start docker
    success "Docker started."
fi

# Check Docker context
CURRENT_CONTEXT=$(docker context show)
if [[ "$CURRENT_CONTEXT" != "default" ]]; then
    error "Docker context is '$CURRENT_CONTEXT', expected 'default'."
    exit 1
fi
success "Docker context is correct: $CURRENT_CONTEXT"

# Parse flags
if [[ "$1" == "--restore" ]]; then
    RESTORE_PIPELINE=true
fi

echo "Running Docker Compose cleanup."
docker compose down || true
docker system prune -af --volumes || true

echo "Building Docker Compose services."
docker compose build --no-cache

echo "Starting Docker Compose services."
docker compose up -d

echo "Waiting for FastAPI to become ready."
sleep 5
for i in {1..20}; do
    if nc -z 127.0.0.1 8000; then
        success "FastAPI service is up."
        break
    fi
    echo "FastAPI not ready yet... ($i/20)"
    sleep 2
done

# Optional pipeline restore
if [ "$RESTORE_PIPELINE" = true ]; then
    echo "Restoring pipeline."
    python scripts/restore_pipeline.py || {
        error "Pipeline restore failed."
        docker compose logs
        docker compose down
        exit 1
    }
else
    echo "Skipping pipeline restore (use --restore to enable)."
fi

echo "Testing FastAPI endpoints."
python scripts/test_app_endpoints.py || {
    error "FastAPI tests failed."
    docker compose logs
    docker compose down
    exit 1
}

echo "Cleaning up Docker Compose services."
docker compose down
docker system prune -af --volumes || true

success "Docker Compose workflow completed successfully."
