import pickle
import joblib
import os
from pathlib import Path

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

def save_model(model, model_name, use_joblib=False):
    """Save a trained model to disk"""
    model_path = MODELS_DIR / f'{model_name.replace(" ", "_")}_model.pkl'
    
    if use_joblib:
        joblib.dump(model, model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model_path

def load_model(model_name, category=None, use_joblib=False):
    """Load a trained model from disk"""
    if category:
        model_path = MODELS_DIR / category / f'{model_name.replace(" ", "_")}_model.pkl'
    else:
        model_path = MODELS_DIR / f'{model_name.replace(" ", "_")}_model.pkl'
    
    if not model_path.exists():
        return None
    
    if use_joblib:
        model = joblib.load(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    return model

def get_available_models():
    """Get list of available trained models"""
    model_files = list(MODELS_DIR.glob('*_model.pkl'))
    models = [f.stem.replace('_model', '').replace('_', ' ') for f in model_files]
    return models


def get_model_info(model_name, category=None, use_joblib=False):
    """Get information about a trained model"""
    if category:
        model_path = MODELS_DIR / category / f'{model_name.replace(" ", "_")}_model.pkl'
    else:
        model_path = MODELS_DIR / f'{model_name.replace(" ", "_")}_model.pkl'
    
    if not model_path.exists():
        return None
    
    # Get file size in MB
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Get model type
    try:
        if use_joblib:
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        model_type = type(model).__name__
    except Exception as e:
        model_type = "Unknown"
    
    return {
        'name': model_name,
        'path': str(model_path),
        'size_mb': file_size_mb,
        'size_bytes': file_size_bytes,
        'type': model_type,
        'exists': True
    }

def delete_model(model_name):
    """Delete a trained model"""
    model_path = MODELS_DIR / f'{model_name.replace(" ", "_")}_model.pkl'
    if model_path.exists():
        os.remove(model_path)
        return True
    return False

def load_scaler(category=None):
    """Load a fitted scaler from disk"""
    if category:
        scaler_path = MODELS_DIR / category / 'scaler.pkl'
    else:
        scaler_path = MODELS_DIR / 'scaler.pkl'
    
    if not scaler_path.exists():
        return None
    
    scaler = joblib.load(scaler_path)

    return scaler