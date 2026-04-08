CATEGORY = "Real Estate"
TITLE = "Real Estate"
DESCRIPTION = "Estimate house prices with the default regressor, inspect it, train your own model, and compare results."
TASK_TYPE = "regression"
USE_JOBLIB = False
DEFAULT_MODELS = ["Linear Regression", "Random Forest"]
TARGET_COLUMN = "price"
RAW_FORM_FIELDS = {
    "year_sold": {"type": "number", "min": 1900, "max": 2100, "default": 2013},
    "property_tax": {"extra": "$","type": "number", "min": 0, "max": 20000, "default": 234},
    "insurance": {"extra": "$","type": "number", "min": 0, "max": 5000, "default": 81},
    "beds": {"type": "number", "min": 0, "max": 20, "default": 3},
    "baths": {"type": "number", "min": 0, "max": 20, "default": 2},
    "sqft": {"type": "number", "min": 0, "max": 20000, "default": 1500},
    "year_built": {"type": "number", "min": 1800, "max": 2100, "default": 1995},
    "lot_size": {"extra": "sqft", "type": "number", "min": 0, "max": 1000000, "default": 5000},
    "basement": {"type": "select", "options": [0, 1], "default": 0},
    "popular": {"type": "select", "options": [0, 1], "default": 0},
    "recession": {"type": "select", "options": [0, 1], "default": 0},
    "property_age": {"extra": "years","type": "number", "min": 0, "max": 300, "default": 20},
    "property_type_Condo": {"type": "select", "options": [0, 1], "default": 0},
}
EXPECTED_FEATURES = list(RAW_FORM_FIELDS.keys())
DEFAULT_TRAINING_MODELS = ["Linear Regression", "Random Forest", "Neural Network Regressor"]
