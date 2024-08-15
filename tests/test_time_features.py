import unittest

import pandas as pd

from src.features.time_features import add_time_features


class TestTimeFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 2],
                "purchase_date": pd.to_datetime(
                    ["2023-01-01", "2023-02-15", "2023-03-10", "2023-04-20"]
                ),
                "purchase_amount": [100, 200, 150, 250],
            }
        )

    def test_add_time_features(self):
        # Applies the function to the dataframe
        featured_df = add_time_features(self.df)

        # Checks if the output is a DataFrame
        self.assertIsInstance(featured_df, pd.DataFrame)

        # Checks if the output has the expected columns
        self.assertIn("year_month", featured_df.columns)
        self.assertIn("year", featured_df.columns)
        self.assertIn("month", featured_df.columns)
        self.assertIn("quarter", featured_df.columns)
        self.assertIn("frequency", featured_df.columns)
        self.assertIn("days_since_last_purchase", featured_df.columns)
        self.assertIn("avg_days_between_purchases", featured_df.columns)

        # Checks if the output has the expected data types
        self.assertEqual(featured_df["year"].dtype, "int")
        self.assertEqual(featured_df["month"].dtype, "int")
        self.assertEqual(featured_df["quarter"].dtype, "int")


if __name__ == "__main__":
    unittest.main()
