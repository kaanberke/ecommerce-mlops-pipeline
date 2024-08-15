import unittest

import pandas as pd

from src.features.customer_features import add_customer_features


class TestCustomerFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 2],
                "purchase_amount": [100, 200, 150, 250],
                "age": [25, 25, 40, 40],
                "gender": ["Male", "Male", "Female", "Female"],
                "annual_income": [50000, 50000, 80000, 80000],
                "year_month": ["2020-01", "2020-02", "2020-01", "2020-02"],
                "avg_days_between_purchases": [10, 10, 15, 15],
                "days_since_last_purchase": [5, 5, 10, 10],
                "frequency": [1, 1, 1, 1],
            }
        )

    def test_add_customer_features(self):
        featured_df = add_customer_features(self.df)

        self.assertIn("clv", featured_df.columns)
        self.assertIn("age_group_19-25", featured_df.columns)
        self.assertIn("age_group_26-35", featured_df.columns)
        self.assertIn("age_group_36-45", featured_df.columns)
        self.assertIn("age_group_46-55", featured_df.columns)
        self.assertIn("age_group_56-65", featured_df.columns)
        self.assertIn("age_group_65+", featured_df.columns)
        self.assertIn("income_group_Low", featured_df.columns)
        self.assertIn("income_group_Medium", featured_df.columns)
        self.assertIn("income_group_High", featured_df.columns)
        self.assertIn("income_group_Very High", featured_df.columns)
        self.assertIn("age_income_interaction", featured_df.columns)

        self.assertEqual(featured_df["clv"].iloc[1], 300)  # 100 + 200
        self.assertEqual(featured_df["clv"].iloc[3], 400)  # 150 + 250


if __name__ == "__main__":
    unittest.main()
