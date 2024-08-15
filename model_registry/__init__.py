from pathlib import Path

current_dir = Path(__file__).resolve().parent

DB_PATH = current_dir / "data" / "model_registry.db"
