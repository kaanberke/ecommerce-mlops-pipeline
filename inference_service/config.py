import os
from pathlib import Path

PROJECT_PATH = Path(os.getcwd())
MODEL_REGISTRY_URL = os.getenv("MODEL_REGISTRY_URL", "http://localhost:8000")
PREPROCESSING_SERVICE_URL = os.getenv(
    "PREPROCESSING_SERVICE_URL", "http://localhost:8001"
)
