# Compatibility layer modeled after the uploaded model_manager.py
import os
import pickle
from pathlib import Path
import joblib

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def save_model(model, model_name, use_joblib=False):
    model_path = MODELS_DIR / f"{model_name.replace(' ', '_')}_model.pkl"
    if use_joblib:
        joblib.dump(model, model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model_path

def load_model(model_name, category=None, use_joblib=False):
    if category:
        model_path = MODELS_DIR / category / f"{model_name.replace(' ', '_')}_model.pkl"
    else:
        model_path = MODELS_DIR / f"{model_name.replace(' ', '_')}_model.pkl"
    if not model_path.exists():
        return None
    if use_joblib:
        return joblib.load(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)

def get_available_models():
    model_files = list(MODELS_DIR.glob("*_model.pkl"))
    return [f.stem.replace("_model", "").replace("_", " ") for f in model_files]

def get_model_info(model_name, category=None, use_joblib=False):
    if category:
        model_path = MODELS_DIR / category / f"{model_name.replace(' ', '_')}_model.pkl"
    else:
        model_path = MODELS_DIR / f"{model_name.replace(' ', '_')}_model.pkl"
    if not model_path.exists():
        return None
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    try:
        model = joblib.load(model_path) if use_joblib else pickle.load(open(model_path, "rb"))
        model_type = type(model).__name__
    except Exception:
        model_type = "Unknown"
    return {
        "name": model_name,
        "path": str(model_path),
        "size_mb": file_size_mb,
        "size_bytes": file_size_bytes,
        "type": model_type,
        "exists": True,
    }

def delete_model(model_name):
    model_path = MODELS_DIR / f"{model_name.replace(' ', '_')}_model.pkl"
    if model_path.exists():
        os.remove(model_path)
        return True
    return False

def load_scaler(category=None):
    scaler_path = MODELS_DIR / category / "scaler.pkl" if category else MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)
