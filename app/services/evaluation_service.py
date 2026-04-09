import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from app.utils.logger import get_logger

logger = get_logger(__name__)
def evaluate_supervised_model(model, X_test, y_test, task_type: str):
    logger.info(f"Evulateting supervised model...")
    predictions = model.predict(X_test)
    if task_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        }
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }

def evaluate_clustering_model(X, labels):
    logger.info(f"Evaluating clustering model...")
    unique_labels = len(set(labels))
    if unique_labels < 2:
        return {"silhouette": 0.0}
    return {"silhouette": round(float(silhouette_score(X, labels)), 4)}

def feature_importance_frame(model, feature_names):
    logger.info(f"Evaluating feature importance...")
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).reset_index(drop=True)
    if hasattr(model, "coef_"):
        coef = model.coef_
        if hasattr(coef, "ndim") and coef.ndim > 1:
            coef = coef[0]
        return pd.DataFrame({"feature": feature_names, "coefficient": coef}).assign(abs_coefficient=lambda d: d["coefficient"].abs()).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return None
