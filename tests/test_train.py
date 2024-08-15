import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from train import (
    evaluate_model,
    load_and_preprocess_data,
    save_model,
    train_random_forest,
    train_xgboost,
)


class DummyModel:
    def __init__(self):
        self.some_attribute = "test"

    def predict(self, X):
        return np.array([1, 2, 3, 4])


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.mock_df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 2],
                "purchase_date": pd.to_datetime(
                    ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"]
                ),
                "purchase_amount": [100, 200, 300, 400],
                "annual_income": [50000, 60000, 70000, 80000],
                "age": [30, 40, 50, 60],
                "gender": ["Male", "Female", "Male", "Female"],
            }
        )
        self.mock_standardization_params = {
            "purchase_amount": {"mean": 250, "std": 150},
            "annual_income": {"mean": 65000, "std": 15000},
            "age": {"mean": 45, "std": 15},
        }

    @patch("train.load_data")
    @patch("train.clean_data")
    @patch("train.add_time_features")
    @patch("train.add_customer_features")
    @patch("train.standardize_features")
    @patch("train.fill_missing_monthly_purchases")
    @patch("train.create_target_variable")
    def test_load_and_preprocess_data(
        self,
        mock_create_target,
        mock_missing_months_filled,
        mock_standardize,
        mock_add_customer,
        mock_add_time,
        mock_clean,
        mock_load,
    ):
        mock_load.return_value = self.mock_df
        mock_clean.return_value = self.mock_df
        mock_add_time.return_value = self.mock_df
        mock_add_customer.return_value = self.mock_df
        mock_standardize.return_value = (self.mock_df, self.mock_standardization_params)
        mock_missing_months_filled.return_value = self.mock_df
        mock_create_target.return_value = self.mock_df

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = Path(tmpdirname) / "test_data.csv"
            result_df, result_params = load_and_preprocess_data(temp_path)

            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_df.shape, self.mock_df.shape)
            self.assertEqual(result_params, self.mock_standardization_params)

        self.assertFalse(Path(tmpdirname).exists())

    @patch("train.RandomForestRegressor")
    @patch("train.RandomizedSearchCV")
    def test_train_random_forest(self, mock_random_search, mock_rf):
        mock_rf.return_value = MagicMock()
        mock_random_search.return_value = MagicMock()

        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.rand(100))

        result = train_random_forest(X_train, y_train)

        self.assertIsInstance(result, MagicMock)
        mock_random_search.assert_called_once()

    @patch("train.XGBRegressor")
    @patch("train.RandomizedSearchCV")
    def test_train_xgboost(self, mock_random_search, mock_xgb):
        mock_xgb.return_value = MagicMock()
        mock_random_search.return_value = MagicMock()

        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.rand(100))

        result = train_xgboost(X_train, y_train)

        self.assertIsInstance(result, MagicMock)
        mock_random_search.assert_called_once()

    def test_evaluate_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 2, 3, 4])

        X_test = pd.DataFrame(np.random.rand(4, 5))
        y_test = pd.Series([1, 2, 3, 4])

        rmse = evaluate_model(
            mock_model, X_test, y_test, self.mock_standardization_params
        )

        self.assertIsInstance(rmse, float)
        mock_model.predict.assert_called_once()

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a real RandomForestRegressor instance
            model = RandomForestRegressor(n_estimators=10, random_state=42)

            # Fit the model with some dummy data
            X = np.random.rand(10, 5)
            y = np.random.rand(10)
            model.fit(X, y)

            output_path = Path(tmpdirname) / "model.pkl"

            # Call the save_model function
            save_model(model, output_path)

            # Verify that the file was created
            self.assertTrue(output_path.exists())

            # Print file size and first few bytes for debugging
            print(f"File size: {output_path.stat().st_size} bytes")
            with open(output_path, "rb") as f:
                print(f"First 20 bytes: {f.read(20)}")

            # Try to load the model using different methods
            try:
                with open(output_path, "rb") as f:
                    loaded_model = pickle.load(f)
                print("Successfully loaded with pickle")
            except Exception as e:
                print(f"Failed to load with pickle: {str(e)}")

            try:
                loaded_model = joblib.load(output_path)
                print("Successfully loaded with joblib")
            except Exception as e:
                print(f"Failed to load with joblib: {str(e)}")

            # If we successfully loaded the model, perform some checks
            if "loaded_model" in locals():
                self.assertIsInstance(loaded_model, RandomForestRegressor)
                self.assertEqual(loaded_model.n_estimators, 10)
                self.assertEqual(loaded_model.random_state, 42)

                # Test that the loaded model makes the same predictions as the original
                X_test = np.random.rand(5, 5)
                np.testing.assert_array_almost_equal(
                    model.predict(X_test), loaded_model.predict(X_test)
                )
            else:
                self.fail("Failed to load the model with both pickle and joblib")


if __name__ == "__main__":
    unittest.main()
