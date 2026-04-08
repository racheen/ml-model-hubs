import pandas as pd
from app.core.schemas import PredictionResult
from app.repositories.model_repository import load_model, load_scaler

def predict_with_default_model(category: str, model_name: str, processed_input: pd.DataFrame, use_joblib: bool = False) -> PredictionResult:
    model = load_model(model_name=model_name, category=category, use_joblib=use_joblib)
    if model is None:
        raise FileNotFoundError(f"Model '{model_name}' not found in category '{category}'.")
    scaler = load_scaler(category)
    inference_input = processed_input.copy()
    if scaler is not None:
        try:
            inference_input = scaler.transform(inference_input)
        except Exception:
            print(f"Error transforming input data using scaler for category '{category}'.")
            pass
        
    prediction = model.predict(inference_input)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(inference_input)[0][1]
        except Exception:
            probabilities = None
    return PredictionResult(
        model_name=model_name,
        prediction=prediction[0] if len(prediction) else None,
        probabilities=probabilities,
        raw_output=prediction,
    )
