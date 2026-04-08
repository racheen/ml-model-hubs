import pandas as pd
from .config import EXPECTED_FEATURES

def preprocess_input(raw_input: dict) -> pd.DataFrame:
    base = {
        "ApplicantIncome": raw_input["ApplicantIncome"],
        "CoapplicantIncome": raw_input["CoapplicantIncome"],
        "LoanAmount": raw_input["LoanAmount"],
        "Loan_Amount_Term": raw_input["Loan_Amount_Term"],
        "Credit_History": raw_input["Credit_History"],
    }
    frame = pd.DataFrame([base])
    cat_frame = pd.DataFrame([{
        "Gender": raw_input["Gender"],
        "Married": raw_input["Married"],
        "Dependents": raw_input["Dependents"],
        "Education": raw_input["Education"],
        "Self_Employed": raw_input["Self_Employed"],
        "Property_Area": raw_input["Property_Area"],
    }])
    encoded = pd.get_dummies(cat_frame, dtype=int)
    merged = pd.concat([frame, encoded], axis=1)
    for feature in EXPECTED_FEATURES:
        if feature not in merged.columns:
            merged[feature] = 0
    return merged[EXPECTED_FEATURES]
