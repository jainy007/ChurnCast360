# Employee Attrition Analysis Report

## Overview
This report summarizes the findings of the employee attrition analysis for Sailfort Motors.

## Data Cleaning & Preparation
- Missing values: Checked
- Duplicates: Checked
- Outliers: Checked
- Column names standardized 

## Exploratory Data Analysis
### Satisfaction Level Distribution
![Satisfaction Level](images/eda_satisfaction_level.png)

### Average Monthly Hours Distribution
![Average Monthly Hours](images/eda_monthly_hours.png)

### Correlation Matrix
![Correlation Matrix](images/correlation_matrix.png)

### Employee Left by Salary Level
![Left by Salary](images/left_by_salary.png)

### Employee Left by Department
![Left by Department](images/left_by_department.png)

## Model Performance
### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve
![ROC Curve](images/roc_curve.png)

## Conclusion & Recommendations
- The model shows excellent performance with an AUC of 0.99.
- Precision: 0.99, Recall: 0.96, Accuracy: 0.99, F1 Score: 0.97.
- HR team can use this model to proactively identify employees at risk of leaving.
- Recommend quarterly retraining of the model.
- Suggested next steps: feature importance analysis, integration with HR systems.

---
_Report auto-generated by pipeline.py_
