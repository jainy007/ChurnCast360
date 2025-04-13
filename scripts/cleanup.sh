#!/bin/bash

set -e
set -o pipefail

SCOPE=$1  # "docker" or "minikube"

if [[ -z "$SCOPE" ]]; then
  echo "No cleanup scope specified. Usage: cleanup.sh --scope [docker|minikube]"
  exit 1
fi

echo "Starting system cleanup for scope: $SCOPE"

# Docker cleanup
if [[ "$SCOPE" == "docker" ]]; then
  echo "Removing stopped Docker containers..."
  docker container prune -f || true

  echo "Removing dangling Docker images..."
  docker image prune -f || true

  echo "Removing unused Docker networks..."
  docker network prune -f || true

  echo "Removing unused Docker volumes..."
  docker volume prune -f || true

  echo "Docker cleanup completed."
fi

# Minikube cleanup
if [[ "$SCOPE" == "minikube" ]]; then
  echo "Cleaning Minikube dangling images..."
  minikube image prune || echo "Minikube image prune skipped (Minikube might not be running)."

  echo "Checking Kubernetes resources..."
  kubectl get pods --all-namespaces || echo "Kubernetes not running, skipping resource check."

  echo "Minikube cleanup completed."
fi

echo "System cleanup completed successfully!"
