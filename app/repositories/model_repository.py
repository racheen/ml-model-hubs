import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from app.config.paths import MODELS_DIR

def _resolve_model_path(model_name: str, category: Optional[str] = None) -> Path:
    if category:
        return MODELS_DIR / category / f"{model_name.replace(' ', '_')}_model.pkl"
    return MODELS_DIR / f"{model_name.replace(' ', '_')}_model.pkl"

def _resolve_metrics_path(model_name: str, category: Optional[str] = None) -> Path:
    if category:
        return MODELS_DIR / category / f"{model_name.replace(' ', '_')}_metrics.pkl"
    return MODELS_DIR / f"{model_name.replace(' ', '_')}_metrics.pkl"

def load_model(model_name: str, category: Optional[str] = None, use_joblib: bool = False):
    model_path = _resolve_model_path(model_name, category)
    if not model_path.exists():
        return None
    if use_joblib:
        return joblib.load(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)

def save_model(model: Any, model_name: str, category: Optional[str] = None, use_joblib: bool = False) -> Path:
    target_dir = MODELS_DIR / category if category else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / f"{model_name.replace(' ', '_')}_model.pkl"
    if use_joblib:
        joblib.dump(model, model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model_path

def load_scaler(category: Optional[str] = None):
    scaler_path = (MODELS_DIR / category / "scaler.pkl") if category else (MODELS_DIR / "scaler.pkl")
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)

def save_scaler(scaler: Any, category: Optional[str] = None) -> Path:
    target_dir = MODELS_DIR / category if category else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = target_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    return scaler_path

def load_feature_names(category: str) -> Optional[List[str]]:
    features_path = MODELS_DIR / category / "features.json"
    if not features_path.exists():
        return None
    try:
        return json.loads(features_path.read_text())
    except Exception:
        return None

def save_feature_names(feature_names: List[str], category: str) -> Path:
    target_dir = MODELS_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    features_path = target_dir / "features.json"
    features_path.write_text(json.dumps(feature_names, indent=2))
    return features_path

def get_model_info(model_name: str, category: Optional[str] = None, use_joblib: bool = False) -> Optional[Dict[str, Any]]:
    model_path = _resolve_model_path(model_name, category)
    if not model_path.exists():
        return None
    size_bytes = model_path.stat().st_size
    model_type = "Unknown"
    try:
        model = load_model(model_name, category, use_joblib)
        model_type = type(model).__name__
    except Exception:
        pass
    return {
        "name": model_name,
        "path": str(model_path),
        "size_mb": round(size_bytes / (1024 * 1024), 4),
        "size_bytes": size_bytes,
        "type": model_type,
        "exists": True,
    }

def load_metrics(model_name: str, category: Optional[str] = None):
    metrics_path = _resolve_metrics_path(model_name, category)
    if not metrics_path.exists():
        return None
    return joblib.load(metrics_path)