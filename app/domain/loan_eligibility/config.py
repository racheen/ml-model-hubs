CATEGORY = "Loan Eligibility"
TITLE = "Loan Eligibility"
DESCRIPTION = "Predict loan approval, inspect the trained classifier, build your own classifier, and compare results."
TASK_TYPE = "classification"
USE_JOBLIB = False
DEFAULT_MODELS = ["Logistic Regression", "Random Forest Classifier"]
TARGET_COLUMN = "Loan_Approved"
RAW_FORM_FIELDS = {
    "Gender": { "type": "select", "options": ["Male", "Female"], "default": "Male"},
    "Married": { "type": "select", "options": ["Yes", "No"], "default": "Yes"},
    "Dependents": {"type": "select", "options": ["0", "1", "2", "3+"], "default": "0"},
    "Education": {"type": "select", "options": ["Graduate", "Not Graduate"], "default": "Graduate"},
    "Self_Employed": {"type": "select", "options": ["Yes", "No"], "default": "No"},
    "ApplicantIncome": {"extra": "$", "type": "number", "min": 0, "max": 200000, "default": 5000},
    "CoapplicantIncome": {"extra": "$", "type": "number", "min": 0, "max": 100000, "default": 0},
    "LoanAmount": {"extra": "1000s", "type": "number", "min": 0, "max": 1000, "default": 128},
    "Loan_Amount_Term": {"extra": "months", "type": "number", "min": 0, "max": 600, "default": 360},
    "Credit_History": {"type": "select", "options": [0, 1], "default": 1},
    "Property_Area": {"type": "select", "options": ["Urban", "Semiurban", "Rural"], "default": "Urban"},
}
EXPECTED_FEATURES = [
    "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History",
    "Gender_Female","Gender_Male","Married_No","Married_Yes","Dependents_0","Dependents_1","Dependents_2","Dependents_3+",
    "Education_Graduate","Education_Not Graduate","Self_Employed_No","Self_Employed_Yes",
    "Property_Area_Rural","Property_Area_Semiurban","Property_Area_Urban",
]
DEFAULT_TRAINING_MODELS = ["Logistic Regression", "Random Forest", "Neural Network"]
