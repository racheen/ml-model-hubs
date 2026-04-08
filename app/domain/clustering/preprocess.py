import pandas as pd
from .config import EXPECTED_FEATURES

def preprocess_input(raw_input: dict) -> pd.DataFrame:
    frame = pd.DataFrame([raw_input])
    for feature in EXPECTED_FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0
    return frame[EXPECTED_FEATURES]
