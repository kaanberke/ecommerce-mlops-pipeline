import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import app

client = TestClient(app)


@pytest.fixture
def sample_raw_data():
    return {
        "customer_id": 1,
        "purchase_date": "2024-04-07T11:25:51.893Z",
        "purchase_amount": 100,
        "annual_income": 150000,
        "age": 42,
        "gender": "Female",
    }


@pytest.fixture
def sample_standardization_params():
    return {
        "params": {
            "purchase_amount": {"mean": 50, "std": 25},
            "annual_income": {"mean": 100000, "std": 50000},
            "clv": {"mean": 1000, "std": 500},
            "age_income_interaction": {"mean": 5000000, "std": 2500000},
        }
    }


@pytest.fixture
def mock_customer_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 1],
            "purchase_date": [
                pd.Timestamp("2024-03-07T11:25:51.893Z"),
                pd.Timestamp("2024-02-07T11:25:51.893Z"),
            ],
            "purchase_amount": [90, 80],
            "annual_income": [150000, 150000],
            "age": [42, 42],
            "gender": ["Female", "Female"],
            "next_month_purchase_amount": [80, 90],
        }
    )


def test_preprocess_success(
    sample_raw_data, sample_standardization_params, mock_customer_data
):
    with patch("pandas.read_csv", return_value=mock_customer_data):
        response = client.post(
            "/preprocess",
            json={
                "raw_data": sample_raw_data,
                "standardization_params": sample_standardization_params,
            },
        )

    assert (
        response.status_code == 200
    ), f"Actual status code: {response.status_code}, Response: {response.json()}"
    processed_data = response.json()

    assert "features" in processed_data
    features = processed_data["features"]

    expected_features = [
        "age",
        "gender",
        "frequency",
        "days_since_last_purchase",
        "avg_days_between_purchases",
        "age_group_19-25",
        "age_group_26-35",
        "age_group_36-45",
        "age_group_46-55",
        "age_group_56-65",
        "age_group_65+",
        "income_group_Low",
        "income_group_Medium",
        "income_group_High",
        "income_group_Very High",
        "purchase_amount_standardized",
        "annual_income_standardized",
        "clv_standardized",
        "age_income_interaction_standardized",
    ]
    for feature in expected_features:
        assert (
            feature in features
        ), f"Expected feature '{feature}' not found in processed data"

    for feature in ["gender"] + [f for f in features if f.startswith("age_group_")]:
        assert features[feature] in [
            0,
            1,
        ], f"One-hot encoded feature '{feature}' should be 0 or 1"

    print(f"Processed features: {features}")


def test_preprocess_invalid_data(sample_standardization_params):
    invalid_data = {
        "customer_id": "invalid",
        "purchase_date": "invalid date",
        "purchase_amount": "invalid",
        "annual_income": "invalid",
        "age": "invalid",
        "gender": 123,
    }

    response = client.post(
        "/preprocess",
        json={
            "raw_data": invalid_data,
            "standardization_params": sample_standardization_params,
        },
    )

    assert response.status_code == 422


def test_preprocess_missing_data(sample_raw_data, sample_standardization_params):
    del sample_raw_data["purchase_amount"]

    response = client.post(
        "/preprocess",
        json={
            "raw_data": sample_raw_data,
            "standardization_params": sample_standardization_params,
        },
    )

    print(response.json())

    assert response.status_code == 422


def test_preprocess_customer_not_found(sample_raw_data, sample_standardization_params):
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        response = client.post(
            "/preprocess",
            json={
                "raw_data": sample_raw_data,
                "standardization_params": sample_standardization_params,
            },
        )

    assert response.status_code == 500
    assert "Preprocessing error" in response.json()["detail"]


@patch("pandas.read_csv")
def test_preprocess_file_not_found_error(
    mock_read_csv, sample_raw_data, sample_standardization_params
):
    mock_read_csv.side_effect = FileNotFoundError("File not found")

    response = client.post(
        "/preprocess",
        json={
            "raw_data": sample_raw_data,
            "standardization_params": sample_standardization_params,
        },
    )

    assert response.status_code == 500
    assert "Preprocessing error" in response.json()["detail"]


def test_preprocess_invalid_standardization_params(sample_raw_data):
    invalid_params = {
        "params": {
            "purchase_amount": {"mean": "invalid", "std": 25},
        }
    }

    response = client.post(
        "/preprocess",
        json={"raw_data": sample_raw_data, "standardization_params": invalid_params},
    )

    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main()
