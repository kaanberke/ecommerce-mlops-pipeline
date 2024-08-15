import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from model_registry.db_models import Base
from model_registry.model_registry import ModelRegistry


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="function")
def model_registry(db_session):
    registry = ModelRegistry()
    registry.db = db_session
    return registry


@pytest.fixture(scope="function")
def sample_model_info():
    return {
        "model_name": "test_model",
        "version": "1.0.0",
        "file_path": "/path/to/model.joblib",
        "standardization_params": {"mean": 0, "std": 1},
        "model_params": {"n_estimators": 100},
        "additional_metadata": {"author": "Test User"},
    }
