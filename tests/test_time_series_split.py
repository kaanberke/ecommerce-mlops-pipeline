import unittest

import pandas as pd

from src.training.time_series_split import time_series_train_test_split


class TestTimeSeriesTrainTestSplit(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.df = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "year_month": [
                    "2020-01",
                    "2020-02",
                    "2020-03",
                    "2021-04",
                    "2021-05",
                    "2021-06",
                    "2022-07",
                    "2022-08",
                    "2022-09",
                ],
                "year": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
                "month": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "gender": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "purchase_amount": [100, 200, 300, 400, 500, 600, 700, 800, 900],
                "next_month_purchase_amount": [
                    1.2,
                    2.1,
                    3.0,
                    4.5,
                    5.7,
                    6.3,
                    7.8,
                    8.9,
                    9.1,
                ],
            }
        )

    def test_split_ratio(self):
        test_size = 0.3
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            self.df, test_size=test_size
        )

        # Check if the split ratio is correct
        self.assertAlmostEqual(len(X_test) / len(self.df), test_size, delta=0.1)

    def test_time_order(self):
        X_train, X_test, y_train, y_test = time_series_train_test_split(self.df)

        train_max_row = X_train.sort_values(["year", "month"]).iloc[-1]
        train_max = f"{train_max_row['year']}-{train_max_row['month']}"
        test_max_row = X_test.sort_values(["year", "month"]).iloc[-1]
        test_max = f"{test_max_row['year']}-{test_max_row['month']}"
        self.assertLessEqual(train_max, test_max)

    def test_feature_target_separation(self):
        X_train, X_test, y_train, y_test = time_series_train_test_split(self.df)

        self.assertNotIn("next_month_purchase_amount", X_train.columns)
        self.assertNotIn("next_month_purchase_amount", X_test.columns)
        self.assertEqual(y_train.name, "next_month_purchase_amount")
        self.assertEqual(y_test.name, "next_month_purchase_amount")

    def test_no_data_leakage(self):
        X_train, X_test, y_train, y_test = time_series_train_test_split(self.df)

        train_indices = X_train.index
        test_indices = X_test.index

        self.assertTrue(set(train_indices).isdisjoint(test_indices))

    def test_output_types(self):
        X_train, X_test, y_train, y_test = time_series_train_test_split(self.df)

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

    def test_error_handling(self):
        # Test with invalid test_size
        with self.assertRaises(AssertionError):
            time_series_train_test_split(self.df, test_size=1.5)

        # Test with missing time column
        df_no_time = self.df.drop("year_month", axis=1)
        with self.assertRaises(AssertionError):
            time_series_train_test_split(df_no_time)

    def test_dropped_columns(self):
        df_with_extra = self.df.copy()
        df_with_extra["purchase_date"] = pd.date_range(start="2021-01-01", periods=9)

        X_train, X_test, y_train, y_test = time_series_train_test_split(df_with_extra)

        self.assertNotIn("purchase_date", X_train.columns)
        self.assertNotIn("year_month", X_train.columns)
        self.assertNotIn("purchase_date", X_test.columns)
        self.assertNotIn("year_month", X_test.columns)


if __name__ == "__main__":
    unittest.main()
