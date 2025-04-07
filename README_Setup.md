## SETUP

### Install Pre-requisites

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Code Quality

We use **Black** and **Ruff** for formatting and linting.

- Auto-fix issues locally:
  ```bash
  pre-commit run --all-files
  ```

### Install Azurite (Azure Blob Simulator)

```bash
sudo npm install -g azurite@3.22.0
```

Run Azurite in a separate terminal:

```bash
azurite --location ./azurite --debug ./azurite/debug.log --loose
```

### Install Azure SDK for Python

```bash
pip install azure-storage-blob
```

---

## PROCESS AND LOAD DATASETS

```bash
python src/ingest/load_datasets.py
```

What it does:
- Loads Kaggle Telco, IBM Telco, Bank Churn, Synthetic Faker data
- Saves raw and cleaned processed files in `/data/`

---

## EDA AND SUMMARISE DATASETS

```bash
python src/eda/run_eda.py
```

What it does:
- Performs Exploratory Data Analysis (EDA)
- Generates summary reports and profiling

---

## FEATURE ENGINEERING

**Goals:**
- Handle nulls, outliers, and type conversions
- Encode categorical variables
- Generate engineered features (e.g., tenure buckets, RFM-like features)
- Save combined `master_dataset.csv` for modeling

Run:

```bash
python src/features/run_feature_engineering.py
```

Outputs:
- Processed master dataset saved to `/data/processed/master_dataset.csv`

---

## DATABASE INTEGRATION (SQLite)

```bash
python src/sql/create_unified_view.py
```

What it does:
- Connects to `churncast360.db`
- Creates unified SQL view `churn_features`
- Prepares data for modeling from SQL

---

## AZURE BLOB SIMULATOR

Upload datasets to Azurite blob storage:

```bash
python src/utils/azure_blob_simulator.py
```

What it does:
- Uploads raw datasets to the local Azure Blob Simulator
- Downloads test blob to verify connectivity
- Lists available blobs for verification

Note: Ensure Azurite is running before executing this script.

---

## BASELINE MODEL TRAINING

**Goals:**
- Load data from SQLite (churn_features view)
- Handle missing values using median imputation
- Encode categorical variables and save encoders
- Train Logistic Regression baseline model
- Evaluate with Accuracy, Recall, and AUC-ROC
- Save trained model and imputer to models/ directory

**Command:**
```bash
python src/model/train_baseline_model.py
```



