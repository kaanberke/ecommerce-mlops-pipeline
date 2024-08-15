import json
import pickle
import uuid

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from model_registry.api import app
from model_registry.db_models import ModelInfoDB, get_db

client = TestClient(app)


def override_get_db():
    try:
        db = next(get_db())
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def db():
    db = next(get_db())
    yield db
    db.close()


@pytest.fixture(scope="module")
def registered_model(db):
    model = RandomForestRegressor(n_estimators=10)
    unique_id = str(uuid.uuid4())[:8]
    model_info = {
        "model_name": f"test_model_{unique_id}",
        "version": "1.0.0",
        "standardization_params": json.dumps({"mean": 0, "std": 1}),
        "model_params": json.dumps({"n_estimators": 10}),
        "additional_metadata": json.dumps({"author": "Test User"}),
    }
    model_bytes = pickle.dumps(model)
    files = {"model_file": ("model.joblib", model_bytes)}
    response = client.post("/register_model", files=files, data=model_info)
    assert response.status_code == 200, f"Failed to register model: {response.text}"
    yield model_info

    db_model_info = (
        db.query(ModelInfoDB)
        .filter(
            ModelInfoDB.model_name == model_info["model_name"],
            ModelInfoDB.version == model_info["version"],
        )
        .first()
    )
    if db_model_info:
        db.delete(db_model_info)
        db.commit()


def test_register_model(registered_model):
    response = client.get(
        f"/get_model_info/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 200
    assert response.json()["model_name"] == registered_model["model_name"]
    assert response.json()["version"] == registered_model["version"]


def test_get_model_info(registered_model):
    response = client.get(
        f"/get_model_info/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 200
    assert response.json()["model_name"] == registered_model["model_name"]
    assert response.json()["version"] == registered_model["version"]


def test_get_latest_model_info(registered_model):
    response = client.get(f"/get_latest_model_info/{registered_model['model_name']}")
    assert response.status_code == 200
    assert response.json()["model_name"] == registered_model["model_name"]


def test_list_models(registered_model):
    response = client.get("/list_models")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert any(
        model["model_name"] == registered_model["model_name"]
        for model in response.json()
    )


def test_get_model(registered_model):
    response = client.get(
        f"/get_model/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


def test_get_nonexistent_model():
    response = client.get("/get_model_info/nonexistent_model/1.0.0")
    assert response.status_code == 404


@pytest.mark.run(order=-1)
def test_remove_model(registered_model):
    response = client.get(
        f"/get_model_info/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 200, "Model should exist before removal"

    response = client.delete(
        f"/remove_model/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 200
    assert (
        f"Model '{registered_model['model_name']}' version '{registered_model['version']}' removed successfully"
        in response.json()["message"]
    )

    response = client.get(
        f"/get_model_info/{registered_model['model_name']}/{registered_model['version']}"
    )
    assert response.status_code == 404, "Model should not exist after removal"


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_models(db):
    yield
    test_models = (
        db.query(ModelInfoDB).filter(ModelInfoDB.model_name.like("test_model_%")).all()
    )
    for model in test_models:
        db.delete(model)
    db.commit()
