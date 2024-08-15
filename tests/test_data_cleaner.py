import unittest

import pandas as pd

from src.preprocessing.data_cleaner import clean_data


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 3, 3],
                "purchase_date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ),
                "purchase_amount": [100, -50, 200, 300, 400],
                "annual_income": [50000, 50000, -1000, 70000, 70000],
                "age": [30, 30, 0, 40, 40],
                "gender": ["M", "M", "F", None, "F"],
                "next_month_purchase_amount": [None, None, None, None, None],
            }
        )

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)

        # Asserts that the cleaned dataframe has the correct shape
        self.assertEqual(len(cleaned_df), 3)

        # Asserts that the cleaned dataframe has the correct columns
        self.assertNotIn("next_month_purchase_amount", cleaned_df.columns)

        # Asserts that the cleaned dataframe has the correct values
        self.assertTrue((cleaned_df["purchase_amount"] >= 0).all())
        self.assertTrue((cleaned_df["annual_income"] >= 0).all())
        self.assertTrue((cleaned_df["age"] > 0).all())
        self.assertFalse(cleaned_df["gender"].isnull().any())


if __name__ == "__main__":
    unittest.main()
