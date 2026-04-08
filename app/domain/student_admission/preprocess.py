import pandas as pd
from .config import EXPECTED_FEATURES

def preprocess_input(raw_input: dict) -> pd.DataFrame:
    numeric = {
        "GRE_Score": raw_input["GRE_Score"],
        "TOEFL_Score": raw_input["TOEFL_Score"],
        "SOP": raw_input["SOP"],
        "LOR": raw_input["LOR"],
        "CGPA": raw_input["CGPA"],
    }
    cat = pd.DataFrame([{
        "University_Rating": raw_input["University_Rating"],
        "Research": raw_input["Research"],
    }]).astype(str)
    encoded = pd.get_dummies(cat, columns=["University_Rating", "Research"], dtype=int)
    frame = pd.DataFrame([numeric])
    merged = pd.concat([frame, encoded], axis=1)
    for feature in EXPECTED_FEATURES:
        if feature not in merged.columns:
            merged[feature] = 0
    return merged[EXPECTED_FEATURES]
