FROM python:3.11-slim

WORKDIR /app

# Copy source code
COPY ./src /app/src
COPY ./models /app/models
COPY ./data /app/data

# Install dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn scikit-learn xgboost joblib pydantic mlflow

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
