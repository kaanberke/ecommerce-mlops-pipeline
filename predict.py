import os
import sys
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import pretty_errors  # noqa: F401
import requests

PROJECT_PATH = Path(os.getcwd())
sys.path.append(str(PROJECT_PATH))

from src.inference.inference import run_inference
from src.preprocessing.data_loader import load_data
from src.utils.logger import create_logger

logger = create_logger()

# Model registry service URL
MODEL_REGISTRY_URL = os.getenv("MODEL_REGISTRY_URL", "http://localhost:8000")
PREPROCESSING_SERVICE_URL = os.getenv(
    "PREPROCESSING_SERVICE_URL", "http://localhost:8001"
)


def get_model_info(model_name, version):
    response = requests.get(
        f"{MODEL_REGISTRY_URL}/get_model_info/{model_name}/{version}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get model info: {response.text}")


def get_model_file(model_name, version):
    response = requests.get(f"{MODEL_REGISTRY_URL}/get_model/{model_name}/{version}")
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get model file: {response.text}")


def main():
    # Paths to the input data
    INPUT_DATA_PATH = PROJECT_PATH / "data" / "test_data" / "customer_purchases.csv"

    # Model names and versions
    MODEL_NAME = "random_forest"
    MODEL_VERSION = "1.0"  # Update with the correct version

    # Loads test data
    test_data = load_data(str(INPUT_DATA_PATH))
    test_data["purchase_date"] = test_data["purchase_date"].dt.strftime(
        "%Y-%m-%dT-%H-%M-%S+03:00"
    )

    if "next_month_purchase_amount" in test_data.columns:
        test_data = test_data.drop(columns=["next_month_purchase_amount"])

    # Gets model info and file
    model_info = get_model_info(MODEL_NAME, MODEL_VERSION)
    model_bytes = get_model_file(MODEL_NAME, MODEL_VERSION)

    # Deserialize the model
    model = joblib.load(BytesIO(model_bytes))

    # Extract standardization parameters from model info
    standardization_params = model_info["standardization_params"]

    results = []
    for _, row in test_data.iterrows():
        response = requests.post(
            f"{PREPROCESSING_SERVICE_URL}/preprocess",
            json={
                "raw_data": row.to_dict(),
                "standardization_params": standardization_params,
            },
        )
        processed_data_df = pd.DataFrame(response.json()["features"], index=[0])

        # Runs inference using the test data, model, and standardization parameters
        _, predictions = run_inference(
            processed_data_df, [model], [standardization_params]
        )
        results.append(predictions[0])
        logger.info(f"Predictions: {predictions[0]:.2f}")

    results = pd.DataFrame(results, columns=["predicted_purchase_amount"])
    results = pd.concat([test_data, results], axis=1)

    # Saves results to a CSV file in the data/predictions directory
    output_path = (
        PROJECT_PATH / "data" / "predictions" / "predicted_customer_purchases.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
