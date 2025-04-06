## SETUP 

1. **Install Prereqs**
```pip install -r requirements.txt```

2. **Process and load Datasets**
``` python src/ingest/load_datasets.py ```

3. **EDA and Summarise Datasets**
```  python src/eda/run_eda.py ```


## FEATURE ENGINEERING

**Goals:**
- Handle nulls, outliers, and type conversions
- Encode categorical variables
- Generate engineered features (e.g., tenure buckets, RFM-like features)
- Save a combined master_dataset.csv for modeling

