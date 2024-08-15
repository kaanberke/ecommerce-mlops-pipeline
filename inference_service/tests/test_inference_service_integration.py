import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import pickle
import unittest

import numpy as np
import requests
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from inference_service.config import MODEL_REGISTRY_URL, PREPROCESSING_SERVICE_URL
from inference_service.main import app as inference_app


class TestIntegrationWithPreprocessing(unittest.TestCase):
    def setUp(self):
        self.inference_client = TestClient(inference_app)
        self.model_registry_url = MODEL_REGISTRY_URL
        self.preprocessing_url = PREPROCESSING_SERVICE_URL

    def test_preprocessing(self):
        raw_data = {
            "customer_id": 1,
            "purchase_date": "2024-04-07T11:25:51.893Z",
            "purchase_amount": 100,
            "annual_income": 150000,
            "age": 42,
            "gender": "Male",
        }
        standardization_params = {
            "params": {
                "purchase_amount": {"mean": 745.0633246786966, "std": 505.881465617238},
                "annual_income": {"mean": 84439.23975056388, "std": 34286.34023794603},
                "clv": {"mean": 2309.004054264573, "std": 1678.1222367214195},
                "age_income_interaction": {
                    "mean": 3203111.5195530728,
                    "std": 1873724.0165341266,
                },
            }
        }

        response = requests.post(
            f"{self.preprocessing_url}/preprocess",
            json={
                "raw_data": raw_data,
                "standardization_params": standardization_params,
            },
        )
        self.assertEqual(
            response.status_code, 200, f"Preprocessing failed: {response.text}"
        )
        processed_data = response.json()
        self.assertIn("features", processed_data)
        self.assertIsInstance(processed_data["features"], dict)

    def test_end_to_end_prediction_with_preprocessing(self):
        dummy_model = RandomForestRegressor(n_estimators=2, random_state=42)
        X = np.random.rand(100, 19)
        y = np.random.rand(100)
        dummy_model.fit(X, y)

        dummy_model_bytes = pickle.dumps(dummy_model)
        model_info = {
            "model_name": "dummy_model",
            "version": "1.0",
            "standardization_params": json.dumps(
                {
                    "purchase_amount": {
                        "mean": 745.0633246786966,
                        "std": 505.881465617238,
                    },
                    "annual_income": {
                        "mean": 84439.23975056388,
                        "std": 34286.34023794603,
                    },
                    "clv": {"mean": 2309.004054264573, "std": 1678.1222367214195},
                    "age_income_interaction": {
                        "mean": 3203111.5195530728,
                        "std": 1873724.0165341266,
                    },
                }
            ),
            "model_params": json.dumps(dummy_model.get_params()),
            "additional_metadata": json.dumps({}),
        }

        files = {
            "model_file": ("model.pkl", dummy_model_bytes, "application/octet-stream")
        }

        response = requests.post(
            f"{self.model_registry_url}/register_model", data=model_info, files=files
        )
        self.assertEqual(
            response.status_code, 200, f"Failed to register model: {response.text}"
        )

        raw_data = {
            "customer_id": 1,
            "purchase_date": "2024-04-07T11:25:51.893Z",
            "purchase_amount": 100,
            "annual_income": 150000,
            "age": 42,
            "gender": "Male",
        }
        standardization_params = json.loads(model_info["standardization_params"])

        preprocess_response = requests.post(
            f"{self.preprocessing_url}/preprocess",
            json={
                "raw_data": raw_data,
                "standardization_params": {"params": standardization_params},
            },
        )
        self.assertEqual(
            preprocess_response.status_code,
            200,
            f"Preprocessing failed: {preprocess_response.text}",
        )
        processed_data = preprocess_response.json()

        prediction_data = {
            "model_name": "dummy_model",
            "model_version": "1.0",
            "purchase_data": raw_data,
            "preprocessed_features": processed_data["features"],
        }
        predict_response = self.inference_client.post("/predict", json=prediction_data)
        self.assertEqual(
            predict_response.status_code,
            200,
            f"Prediction failed: {predict_response.text}",
        )
        prediction = predict_response.json()
        self.assertIn("predicted_purchase_amount", prediction)
        self.assertIsInstance(prediction["predicted_purchase_amount"], (int, float))

    def test_list_models_integration(self):
        response = self.inference_client.get("/models")
        self.assertEqual(
            response.status_code, 200, f"Failed to list models: {response.text}"
        )
        models = response.json()
        self.assertIsInstance(models, list)
        self.assertTrue(
            any(model["model_name"] == "dummy_model" for model in models),
            "Dummy model not found in the list of models",
        )

        response = requests.delete(
            f"{self.model_registry_url}/remove_model/dummy_model/1.0"
        )
        self.assertEqual(
            response.status_code, 200, f"Failed to remove dummy model: {response.text}"
        )


if __name__ == "__main__":
    unittest.main()
