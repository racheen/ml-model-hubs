CATEGORY = "Customer Segmentation"
TITLE = "Customer Segmentation"
DESCRIPTION = "Assign a customer to a cluster with KMeans, inspect the saved clustering model, build your own clustering solution, and compare labels."
TASK_TYPE = "clustering"
USE_JOBLIB = False
DEFAULT_MODELS = ["KMeans"]
TARGET_COLUMN = None
RAW_FORM_FIELDS = {
    "Age": {"type": "number", "min": 0, "max": 100, "default": 30},
    "Annual_Income": {"type": "number", "min": 0, "max": 1000, "default": 60},
    "Spending_Score": {"type": "number", "min": 0, "max": 100, "default": 50},
}
EXPECTED_FEATURES = ["Age", "Annual_Income", "Spending_Score"]
DEFAULT_TRAINING_MODELS = ["KMeans"]
