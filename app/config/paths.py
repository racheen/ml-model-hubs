from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
REPORTS_DIR = ROOT_DIR / "reports"

for path in [MODELS_DIR, DATA_DIR, ASSETS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
