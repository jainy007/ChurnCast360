## ChurnCast360 - Daily Targets Tracker (Revised)

**Purpose:**  
Track daily targets for your 14-day sprint toward KPMG Manager - Data Science interview readiness with ChurnCast360 project.

**Project Summary**

ChurnCast360 is an enterprise-grade customer churn prediction toolkit, simulating Azure-scale MLOps pipeline with:
- **Python**, **FastAPI**, **Azure Blob Simulator** (Azurite)
- **MLflow** for experiment tracking
- **Docker** for containerization
- **Minikube** for local Kubernetes orchestration
- **GitHub Actions** with self-hosted runner for CI/CD automation
- Multi-dataset, workspace-based, no notebooks.  
- *Completely interview ready.*

### Daily Targets

### Day 1: Project Setup + Data Ingestion

- Project folder, venv, structure
- Load Kaggle Telco, IBM Telco, Bank Churn, Synthetic Faker data
- Save raw + processed data
- Completed

### Day 2: Feature Engineering

- Clean data: nulls, outliers
- Generate new features: tenure buckets, spend per ticket
- Save master dataset
- Completed

### Day 3: SQLite Integration

- Simulate Azure Blob storage with SQLite (temporary step)
- Create unified SQL view (churn_features)
- Auto-generate SQL view
- Completed

### Day 4: Azure Blob (Simulated) & Clean GitHub

- Azure Blob simulator (Azurite) integration
- Move processed datasets to blob storage
- Clean GitHub repo (remove LFS hits, clean history)
- Completed

### Day 5: Baseline Model

- Logistic Regression model
- Train on churn_features
- Evaluate: Accuracy, Recall, AUC-ROC
- Save model `.pkl`
- Completed

### Day 6: Model Tuning

- RandomForest, XGBoost
- Hyperparameter tuning
- Evaluation comparison

### Day 7: MLflow Tracking

- MLflow experiments, params, metrics
- Run comparisons in MLflow UI (local dev)

### Day 8: Kubernetes Orchestration (Minikube)

- Install & configure Minikube
- Containerize MLflow & model serving (FastAPI)
- Test Minikube local orchestration

### Day 9: CI/CD with GitHub Actions

- Setup self-hosted runner (your PC)
- Auto trigger builds on push
- Run tests, container builds in pipeline

### Day 10: FastAPI Serving

- Build FastAPI service with `/predict` and `/health` endpoints
- Test locally with sample payload
- Containerize and deploy via Minikube

### Day 11: Streamlit Dashboard

- Streamlit dashboard with file upload
- Prediction display
- Feature importance visuals
- Containerize if time permits

### Day 12: Documentation

- Complete README.md
- Architecture diagram (draw.io / Lucidchart)
- Folder explanations, flow diagrams

### Day 13: Metrics Summary + Slide Deck

- PDF Summary (1-pager or 3 slides)
- Business impact, risks, improvements
- Use for mock interview or final presentation

### Day 14: Mock Review + Expansion Ideas

- Practice end-to-end walkthrough
- List expansion ideas: retraining, realtime logging, BERT explanations
- Optional: Certification enrollments (Kubernetes, MLflow, Docker)

---

### Optional (Stretch Goals):
- [ ] Certification: Kubernetes for Developers (CKAD - free auditing mode)
- [ ] Certification: MLflow Fundamentals (Databricks free tier)
- [ ] Container registry: Azure ACR (free tier)
- [ ] Optional cloud deploy (AKS Free Tier + Minikube pipeline)

---

**Total:** 14 days  
**Goal:** Interview-ready, portfolio-quality MLOps project!

