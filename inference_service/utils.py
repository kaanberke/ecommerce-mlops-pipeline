import time
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Tuple

import joblib
import requests
from fastapi import HTTPException

from .config import MODEL_REGISTRY_URL

# Cache for loaded models by model name and version
model_cache: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = {}


def get_model_info(model_name: str, version: str):
    """
    Retrieves model information from the model registry
    Args:
        model_name (str): Name of the model
        version (str): Version of the model
    Returns:
        dict: Model information
    Examples:
        >>> get_model_info("random_forest", "1.0")
        {"model_name": "random_forest", "version": "1.0", "standardization_params": {"param1": "value1"}}
    """
    response = requests.get(
        f"{MODEL_REGISTRY_URL}/get_model_info/{model_name}/{version}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} version {version} not found"
        )


def get_model_file(model_name: str, version: str):
    """
    Retrieves the model file from the model registry
    Args:
        model_name (str): Name of the model
        version (str): Version of the model
    Returns:
        bytes: Model file content
    Examples:
        >>> get_model_file("random_forest", "1.0")
        b"model content"
    """
    response = requests.get(f"{MODEL_REGISTRY_URL}/get_model/{model_name}/{version}")
    if response.status_code == 200:
        return response.content
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model file for {model_name} version {version} not found",
        )


# Caches the 10 most recent model info responses
@lru_cache(maxsize=10)
def get_cached_model_info(model_name: str, version: str):
    return get_model_info(model_name, version)


def load_model(model_name: str, version: str):
    """
    Loads a model from the model registry
    Args:
        model_name (str): Name of the model
        version (str): Version of the model
    Returns:
        Tuple[Any, Dict[str, Any]]: Loaded model and standardization parameters
    Examples:
        >>> load_model("random_forest", "1.0")
        (RandomForestRegressor(), {"param1": "value1"})
    """
    # Defines a cache key for the model based on the model name and version
    cache_key = (model_name, version)

    start_time = time.perf_counter()

    # Checks if the model is already loaded
    if cache_key in model_cache:
        model = model_cache[cache_key]
        print(
            f"Model {model_name} version {version} loaded from cache in {time.perf_counter() - start_time:.6f} seconds"
        )
        return model

    # Retrieves the model information and file from the model registry
    model_info = get_cached_model_info(model_name, version)
    model_bytes = get_model_file(model_name, version)
    print(
        f"Model {model_name} version {version} loaded from registry in {time.perf_counter() - start_time:.6f} seconds"
    )

    # Deserializes the model
    model = joblib.load(BytesIO(model_bytes))
    result = (model, model_info["standardization_params"])

    # Cache the loaded model and standardization parameters
    model_cache[cache_key] = result

    return result


def clear_model_cache():
    """
    Clears the model cache
    """
    global model_cache
    model_cache.clear()
    get_cached_model_info.cache_clear()
    print("Model cache cleared")


def list_available_models():
    """
    Retrieves the list of available models from the model registry
    Returns:
        List[dict]: List of available models
    Examples:
        >>> list_available_models()
        [
            {"model_name": "random_forest", "versions": ["1.0", "2.0"]},
            {"model_name": "linear_regression", "versions": ["1.0"]}
        ]
    """
    response = requests.get(f"{MODEL_REGISTRY_URL}/list_models")
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail="Failed to retrieve list of models")
