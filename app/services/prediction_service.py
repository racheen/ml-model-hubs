import pandas as pd
from app.core.schemas import PredictionResult
from app.repositories.model_repository import load_model, load_scaler
from app.utils.logger import get_logger

logger = get_logger(__name__)

def predict_with_default_model(category: str, model_name: str, processed_input: pd.DataFrame, use_joblib: bool = False) -> PredictionResult:
    logger.info(f"Running prediction using model: {model_name}")
    model = load_model(model_name=model_name, category=category, use_joblib=use_joblib)
    if model is None:
        logger.error(f"Model '{model_name}' not found in category '{category}'.")
        raise FileNotFoundError(f"Model '{model_name}' not found in category '{category}'.")
    scaler = load_scaler(category)
    inference_input = processed_input.copy()
    if scaler is not None:
        try:
            inference_input = scaler.transform(inference_input)
        except Exception as e:
            logger.error(f"Error transforming input data using scaler for category '{category}'.")
            pass
        
    prediction = model.predict(inference_input)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(inference_input)[0][1]
        except Exception:
            logger.error(f"Error calculating probabilities for prediction using model '{model_name}'.")
            probabilities = None
    return PredictionResult(
        model_name=model_name,
        prediction=prediction[0] if len(prediction) else None,
        probabilities=probabilities,
        raw_output=prediction,
    )
