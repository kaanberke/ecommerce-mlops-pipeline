import json
import os
import unittest

import numpy as np
import pandas as pd

from src.features.numerical_features import (
    create_target_variable,
    save_standardization_params,
    standardize_column,
    standardize_features,
)


class TestNumericalFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 2],
                "purchase_amount": [100, 200, 300, 400],
                "annual_income": [50000, 60000, 70000, 80000],
                "clv": [1000, 2000, 3000, 4000],
                "age_income_interaction": [500000, 600000, 700000, 800000],
            }
        )
        self.test_params_path = "test_standardization_params.json"

    def test_standardize_column(self):
        standardized, params = standardize_column(self.df, "purchase_amount")

        self.assertIsInstance(standardized, np.ndarray)
        self.assertIsInstance(params, dict)
        self.assertIn("mean", params)
        self.assertIn("std", params)

        # Checks if standardization is correct
        expected = np.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079])
        np.testing.assert_array_almost_equal(standardized, expected, decimal=6)

    def test_standardize_features(self):
        standardized_df, params = standardize_features(self.df, mode="train")

        self.assertIsInstance(standardized_df, pd.DataFrame)
        self.assertIsInstance(params, dict)

        expected_columns = [
            "customer_id",
            "purchase_amount_standardized",
            "annual_income_standardized",
            "clv_standardized",
            "age_income_interaction_standardized",
        ]
        self.assertListEqual(list(standardized_df.columns), expected_columns)

        for col in [
            "purchase_amount",
            "annual_income",
            "clv",
            "age_income_interaction",
        ]:
            self.assertIn(col, params)
            self.assertIn("mean", params[col])
            self.assertIn("std", params[col])

    def test_create_target_variable(self):
        self.df["purchase_amount_standardized"] = self.df["purchase_amount"]
        df_with_target = create_target_variable(self.df)

        self.assertIn("next_month_purchase_amount", df_with_target.columns)
        self.assertEqual(len(df_with_target), 2)

        expected_target = [200, 400]
        np.testing.assert_array_equal(
            df_with_target["next_month_purchase_amount"].values, expected_target
        )

    def test_save_standardization_params(self):
        params = {
            "purchase_amount": {"mean": 250, "std": 129.09944},
            "annual_income": {"mean": 65000, "std": 12909.944},
        }

        save_standardization_params(params, self.test_params_path)

        self.assertTrue(os.path.exists(self.test_params_path))

        with open(self.test_params_path, "r") as f:
            loaded_params = json.load(f)

        self.assertEqual(params, loaded_params)

    def tearDown(self):
        if os.path.exists(self.test_params_path):
            os.remove(self.test_params_path)


if __name__ == "__main__":
    unittest.main()
