from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class PredictionResult:
    model_name: str
    prediction: Any
    probabilities: Optional[List[float]] = None
    raw_output: Optional[Any] = None

@dataclass
class TrainingResult:
    model_name: str
    estimator: Any
    task_type: str
    metrics: Dict[str, Any]
    feature_names: List[str]
    scaler: Any = None

@dataclass
class ComparisonResult:
    summary_table: pd.DataFrame
    prediction_table: Optional[pd.DataFrame] = None
