"""Project-wide constants for the customer churn pipeline.

Keeping paths, column lists and hyper-parameters in one place makes the
pipeline easy to extend: adding a categorical feature, for example, only
requires appending its name to ``CAT_COLUMNS``.

Author: Colin Wuest
Date created: 2026-11-19
"""

import os

# --------------------------------------------------------------------------- paths
DATA_PATH = os.path.join(".", "data", "bank_data.csv")
EDA_DIR = os.path.join(".", "images", "eda")
RESULTS_DIR = os.path.join(".", "images", "results")
MODELS_DIR = os.path.join(".", "models")
LOGS_DIR = os.path.join(".", "logs")
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")

# --------------------------------------------------------------------------- target
ATTRITION_COLUMN = "Attrition_Flag"
EXISTING_CUSTOMER = "Existing Customer"
RESPONSE = "Churn"

# --------------------------------------------------------------------------- features
CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# --------------------------------------------------------------------------- modelling
TEST_SIZE = 0.3
RANDOM_STATE = 42
LOGISTIC_MAX_ITER = 3000
RF_SEARCH_ITERATIONS = 12
RF_SEARCH_CV_FOLDS = 3
RF_PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_features": ["sqrt"],
    "max_depth": [5, 8, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "criterion": ["gini"],
}

# --------------------------------------------------------------------------- artifacts
EDA_IMAGES = {
    "churn": os.path.join(EDA_DIR, "churn_distribution.png"),
    "customer_age": os.path.join(EDA_DIR, "customer_age_distribution.png"),
    "marital_status": os.path.join(EDA_DIR, "marital_status_distribution.png"),
    "total_trans_ct": os.path.join(EDA_DIR, "total_transaction_distribution.png"),
    "heatmap": os.path.join(EDA_DIR, "heatmap.png"),
}

RESULT_IMAGES = {
    "rf_report": os.path.join(RESULTS_DIR, "rf_results.png"),
    "lr_report": os.path.join(RESULTS_DIR, "logistic_results.png"),
    "roc": os.path.join(RESULTS_DIR, "roc_curve_result.png"),
    "feature_importance": os.path.join(RESULTS_DIR, "feature_importances.png"),
    "shap": os.path.join(RESULTS_DIR, "shap_summary.png"),
}

MODEL_FILES = {
    "rfc": os.path.join(MODELS_DIR, "rfc_model.pkl"),
    "logistic": os.path.join(MODELS_DIR, "logistic_model.pkl"),
}
