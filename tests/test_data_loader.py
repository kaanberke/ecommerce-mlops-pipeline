import unittest

import pandas as pd

from src.preprocessing.data_loader import load_data


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_csv_path = "test_data.csv"
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "purchase_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "purchase_amount": [100, 200, 300],
            }
        )
        df.to_csv(self.test_csv_path, index=False)

    def test_load_data(self):
        df = load_data(self.test_csv_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIsInstance(df["purchase_date"].iloc[0], pd.Timestamp)

    def tearDown(self):
        import os

        os.remove(self.test_csv_path)


if __name__ == "__main__":
    unittest.main()
