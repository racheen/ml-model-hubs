import pandas as pd
from app.core.schemas import ComparisonResult

def compare_metric_sets(default_metrics: dict, custom_metrics: dict) -> ComparisonResult:
    summary = pd.DataFrame([
        {"model_source": "Default Model", **default_metrics},
        {"model_source": "Custom Model", **custom_metrics},
    ])
    return ComparisonResult(summary_table=summary)

def compare_predictions(default_prediction, custom_prediction) -> pd.DataFrame:
    return pd.DataFrame([
        {"model_source": "Default Model", "prediction": default_prediction},
        {"model_source": "Custom Model", "prediction": custom_prediction},
    ])
