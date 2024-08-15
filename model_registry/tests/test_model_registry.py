import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

from model_registry.model_registry import ModelInfo


def test_register_model(model_registry, sample_model_info):
    model_info = ModelInfo(**sample_model_info)
    registered_model = model_registry.register_model(model_info)
    assert registered_model.model_name == sample_model_info["model_name"]
    assert registered_model.version == sample_model_info["version"]


def test_get_model_info(model_registry, sample_model_info):
    model_info = ModelInfo(**sample_model_info)
    model_registry.register_model(model_info)
    retrieved_model = model_registry.get_model_info(
        sample_model_info["model_name"], sample_model_info["version"]
    )
    assert retrieved_model.dict() == model_info.dict()


def test_get_latest_model_info(model_registry, sample_model_info):
    model_info_v1 = ModelInfo(**sample_model_info)
    model_registry.register_model(model_info_v1)

    sample_model_info["version"] = "2.0.0"
    model_info_v2 = ModelInfo(**sample_model_info)
    model_registry.register_model(model_info_v2)

    latest_model = model_registry.get_latest_model_info(sample_model_info["model_name"])
    assert latest_model.version == "2.0.0"


def test_list_models(model_registry, sample_model_info):
    model_info = ModelInfo(**sample_model_info)
    model_registry.register_model(model_info)

    models_list = model_registry.list_models()
    assert len(models_list) == 1
    assert models_list[0]["model_name"] == sample_model_info["model_name"]
    assert models_list[0]["versions"] == [sample_model_info["version"]]


def test_register_duplicate_model(model_registry, sample_model_info):
    model_info = ModelInfo(**sample_model_info)
    model_registry.register_model(model_info)

    with pytest.raises(ValueError, match="already exists in the registry"):
        model_registry.register_model(model_info)


def test_get_nonexistent_model(model_registry):
    with pytest.raises(ValueError, match="not found in registry"):
        model_registry.get_model_info("nonexistent_model", "1.0.0")
