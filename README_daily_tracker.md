## ChurnCast360 - Daily Targets Tracker

Purpose: Track daily targets for your 14-day sprint toward KPMG Manager - Data Science interview readiness with ChurnCast360 project.

### Project Summary

ChurnCast360 is an enterprise-grade customer churn prediction toolkit, simulating Azure-scale MLOps pipeline with Python, FastAPI, Azure (simulated blob), GitHub Actions, and CI/CD practices. Multi-dataset, workspace-based, no notebooks. Interview ready.

### Daily Targets

**Day 1**

#### Project Setup + Data Ingestion


- Project folder, venv, structure
- Load Kaggle Telco, IBM Telco, Bank Churn, Synthetic Faker data
- Save raw + processed data


- Completed

**Day 2**

#### Feature Engineering


- Clean data: nulls, outliers
- Generate new features: tenure buckets, spend per ticket
- Save master dataset


- Completed

**Day 3**

#### SQLite Integration


- Simulate Azure Blob storage with SQLite
- Create unified SQL view (churn_features)
- Auto-generate SQL view


- Completed

**Day 4**

#### Baseline Model


- Logistic Regression model
- Train on churn_features
- Evaluate: Accuracy, Recall, AUC-ROC
- Save model .pkl


- In Progress

**Day 5**

#### Model Tuning


- RandomForest, XGBoost
- Hyperparameter tuning
- Evaluation comparison

**Day 6**

#### MLflow Tracking


- MLflow experiments, params, metrics
- Run comparisons in MLflow UI

**Day 7**

#### Test Automation


- Add Pytest unit tests
- GitHub Actions: auto-test on push
- Optional: Black formatting, linting

**Day 8**

#### Model Registry


- Simulate Azure ML or MLflow registry
- Model versioning logic

**Day 9**

🌐 FastAPI Serving


- Build FastAPI service with /predict and /health
- Test locally with sample payload

**Day 10**

#### Streamlit Dashboard


- Streamlit dashboard with file upload
- Prediction display
- Feature importance visuals

**Day 11**

#### Optional Docker & Deploy


- Dockerize FastAPI
- Simulate deploy (or local uvicorn deploy)

**Day 12**

#### Documentation


- Complete README.md
- Architecture diagram (draw.io)
- Folder explanations

**Day 13**

#### Metrics Summary + Deck


- PDF Summary (1-pager or 3 slides)
- Business impact, risks, improvements

**Day 14**

#### Mock Review + Expansion Ideas


- Practice end-to-end walkthrough
- List expansion ideas: retraining, realtime logging, BERT explanations

