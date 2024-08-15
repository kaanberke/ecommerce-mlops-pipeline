from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.inference.inference import ModelInference, run_inference


@pytest.fixture
def sample_inference_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 2],
            "age": [30, 40],
            "gender": ["Male", "Female"],
            "purchase_date": ["2021-01-01", "2021-01-02"],
            "year_month": [202101, 202101],
            "year": [2021, 2021],
            "month": [1, 1],
            "quarter": [1, 1],
            "frequency": [1, 1],
            "days_since_last_purchase": [0, 0],
            "avg_days_between_purchases": [0, 0],
            "age_group_19-25": [0, 0],
            "age_group_26-35": [1, 0],
            "age_group_36-45": [0, 1],
            "age_group_46-55": [0, 0],
            "age_group_56-65": [0, 0],
            "age_group_65+": [0, 0],
            "income_group_Low": [1, 0],
            "income_group_Medium": [0, 1],
            "income_group_High": [0, 0],
            "income_group_Very High": [0, 0],
            "purchase_amount_standardized": [0.5, 1.0],
            "annual_income_standardized": [0.0, 0.5],
            "clv_standardized": [0.5, 1.0],
            "age_income_interaction_standardized": [0.0, 0.5],
        }
    )


@pytest.fixture
def mock_model():
    model = Mock()
    model.predict.return_value = np.array([0.5, 1.0])
    return model


@pytest.fixture
def mock_standardization_params():
    return {
        "purchase_amount": {"mean": 100, "std": 50},
        "annual_income": {"mean": 55000, "std": 10000},
        "clv": {"mean": 1000, "std": 500},
        "age_income_interaction": {"mean": 1500000, "std": 750000},
    }


class TestModelInference:
    @patch("builtins.open", new_callable=mock_open, read_data="dummy data")
    @patch("src.inference.inference.pickle.load")
    @patch("src.inference.inference.json.load")
    def test_make_predictions(
        self,
        mock_json_load,
        mock_pickle_load,
        mock_file,
        mock_model,
        mock_standardization_params,
        sample_inference_data,
    ):
        mock_pickle_load.return_value = mock_model
        mock_json_load.return_value = mock_standardization_params

        inference = ModelInference(
            model=mock_model, standardization_params=mock_standardization_params
        )
        predictions = inference.make_predictions(sample_inference_data)

        # Calculate expected predictions after reversing standardization
        expected_predictions = (
            np.array([0.5, 1.0]) * mock_standardization_params["purchase_amount"]["std"]
            + mock_standardization_params["purchase_amount"]["mean"]
        )

        np.testing.assert_array_almost_equal(predictions, expected_predictions)
        mock_model.predict.assert_called_once()

        # Check that the model is called with the correct standardized features
        args, _ = mock_model.predict.call_args

        expected_features = [
            "purchase_amount_standardized",
            "annual_income_standardized",
            "clv_standardized",
            "age_income_interaction_standardized",
        ]
        assert list(args[0].columns[-4:]) == expected_features
        assert args[0].shape == (2, 25)  # 2 rows, 4 features


def test_run_inference(sample_inference_data):
    with patch("src.inference.inference.EnsembleInference") as MockEnsembleInference:
        mock_ensemble = Mock()
        mock_ensemble.preprocess_data.return_value = [
            sample_inference_data.copy(),
            sample_inference_data.copy(),
        ]
        mock_ensemble.make_predictions.return_value = np.array([150, 250])
        MockEnsembleInference.return_value = mock_ensemble

        model_paths = [Path("model1.pkl"), Path("model2.pkl")]
        params_paths = [Path("params1.json"), Path("params2.json")]

        result, predictions = run_inference(
            sample_inference_data, model_paths, params_paths
        )

        assert "predicted_purchase_amount" in result.columns
        np.testing.assert_array_almost_equal(predictions, np.array([150, 250]))
        mock_ensemble.make_predictions.assert_called_once()
