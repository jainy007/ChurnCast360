FROM python:3.11-slim

# Install MLflow
RUN pip install mlflow

# Expose MLflow default port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]

# Create directory for artifacts
RUN mkdir -p /mlflow/artifacts
