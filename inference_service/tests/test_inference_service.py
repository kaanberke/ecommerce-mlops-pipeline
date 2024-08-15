import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from inference_service.config import MODEL_REGISTRY_URL
from inference_service.main import app
from inference_service.utils import load_model


class TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("inference_service.main.load_model")
    @patch("inference_service.main.run_inference")
    @patch("inference_service.main.preprocess_data")
    def test_predict_endpoint(
        self, mock_preprocess_data, mock_run_inference, mock_load_model
    ):
        mock_model = MagicMock()
        mock_standardization_params = {
            "purchase_amount": {"mean": 100, "std": 50},
            "annual_income": {"mean": 50000, "std": 10000},
            "clv": {"mean": 1000, "std": 500},
            "age_income_interaction": {"mean": 1500000, "std": 750000},
        }
        mock_load_model.return_value = (mock_model, mock_standardization_params)

        mock_preprocess_data.return_value = {
            "features": {"feature1": 1.0, "feature2": 2.0}
        }

        mock_run_inference.return_value = (
            pd.DataFrame({"predicted_purchase_amount": [100.0]}),
            [100.0],
        )

        test_data = {
            "model_name": "xgboost",
            "model_version": "1.0",
            "purchase_data": {
                "customer_id": 1,
                "purchase_date": "2023-01-01T00:00:00",
                "purchase_amount": 50.0,
                "annual_income": 50000.0,
                "age": 30,
                "gender": "Male",
            },
        }

        print("Test data:", test_data)

        response = self.client.post("/predict", json=test_data)
        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())

        if response.status_code != 200:
            print("Error details:", response.text)

        self.assertEqual(response.status_code, 200)
        self.assertIn("customer_id", response.json())
        self.assertIn("purchase_date", response.json())
        self.assertIn("predicted_purchase_amount", response.json())

        mock_load_model.assert_called_once_with("xgboost", "1.0")
        mock_preprocess_data.assert_called_once()
        mock_run_inference.assert_called_once()

    @patch("inference_service.utils.requests.get")
    def test_list_available_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"model_name": "model1", "versions": ["1.0", "2.0"]},
            {"model_name": "model2", "versions": ["1.0"]},
        ]
        mock_get.return_value = mock_response

        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {"model_name": "model1", "versions": ["1.0", "2.0"]},
                {"model_name": "model2", "versions": ["1.0"]},
            ],
        )

        mock_get.assert_called_once_with(f"{MODEL_REGISTRY_URL}/list_models")

    @patch("inference_service.utils.requests.get")
    def test_load_model(self, mock_get):
        mock_info_response = MagicMock()
        mock_info_response.status_code = 200
        mock_info_response.json.return_value = {
            "model_name": "test_model",
            "version": "1.0",
            "standardization_params": {
                "params": {
                    "purchase_amount": {"mean": 1, "std": 1},
                    "annual_income": {"mean": 1, "std": 1},
                    "clv": {"mean": 1, "std": 1},
                    "age_income_interaction": {"mean": 1, "std": 1},
                }
            },
        }

        mock_file_response = MagicMock()
        mock_file_response.status_code = 200
        mock_file_response.content = b"mock model content"

        mock_get.side_effect = [mock_info_response, mock_file_response]

        with patch("inference_service.utils.joblib.load") as mock_joblib_load:
            mock_model = MagicMock()
            mock_joblib_load.return_value = mock_model
            model, params = load_model("test_model", "1.0")

        self.assertEqual(model, mock_model)
        self.assertEqual(
            params,
            {
                "params": {
                    "purchase_amount": {"mean": 1, "std": 1},
                    "annual_income": {"mean": 1, "std": 1},
                    "clv": {"mean": 1, "std": 1},
                    "age_income_interaction": {"mean": 1, "std": 1},
                }
            },
        )

        mock_get.assert_any_call(f"{MODEL_REGISTRY_URL}/get_model_info/test_model/1.0")
        mock_get.assert_any_call(f"{MODEL_REGISTRY_URL}/get_model/test_model/1.0")


if __name__ == "__main__":
    unittest.main()
