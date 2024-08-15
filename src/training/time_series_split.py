import pandas as pd


def time_series_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = "year_month",
):
    """
    Performs a time series train-test split
    The most recent data for each customer is used as the test set.
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of the data to include in the test split
        time_col (str): Name of the column representing time periods
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    Examples:
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 1, 1, 2, 2, 2],
        ...     "year_month": [202101, 202102, 202103, 202101, 202102, 202103],
        ...     "purchase_amount": [100, 200, 300, 400, 500, 600],
        ...     "next_month_purchase_amount": [1.2, 2.1, 0.3, 0.5, 1.7, 2.3],
        ... })
        >>> X_train, X_test, y_train, y_test = time_series_train_test_split(df)
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(test_size, float), "test_size must be a float"
    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    assert time_col in df.columns, f"{time_col} not found in dataframe columns"

    # Sorts the dataframe by customer_id and time column
    df = df.sort_values(["customer_id", time_col])

    # Drops purchase_date column if it exists
    if "purchase_date" in df.columns:
        df = df.drop("purchase_date", axis=1)

    # Drops year_month column if it exists
    if "year_month" in df.columns:
        df = df.drop("year_month", axis=1)

    # Identifies the split point for each customer
    split_index = (
        df.groupby("customer_id")
        .apply(lambda x: int(len(x) * (1 - test_size)), include_groups=False)
        .reset_index()
    )
    split_index.columns = ["customer_id", "split_index"]

    # Merges the split index back to the main dataframe
    df = pd.merge(df, split_index, on="customer_id")

    # Creates a boolean mask for the test set
    df["is_test"] = df.groupby("customer_id").cumcount() >= df["split_index"]

    # Drops the customer_id column
    df = df.drop("customer_id", axis=1)

    # Splits the data
    train = df[~df["is_test"]].drop(["split_index", "is_test"], axis=1)
    test = df[df["is_test"]].drop(["split_index", "is_test"], axis=1)

    # Separates features and target
    X_train = train.drop("next_month_purchase_amount", axis=1)
    y_train = train["next_month_purchase_amount"]

    X_test = test.drop("next_month_purchase_amount", axis=1)
    y_test = test["next_month_purchase_amount"]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.features.customer_features import add_customer_features
    from src.features.numerical_features import (
        create_target_variable,
        standardize_features,
    )
    from src.features.time_features import add_time_features
    from src.preprocessing.data_cleaner import clean_data
    from src.preprocessing.data_loader import load_data

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )

    OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/processed/"

    df = load_data(DATA_PATH)
    cleaned_df = clean_data(df)
    time_featured_df = add_time_features(cleaned_df)
    customer_featured_df = add_customer_features(time_featured_df)
    standardized_df, standardization_params = standardize_features(customer_featured_df)
    final_df = create_target_variable(standardized_df)
    print(final_df.head())

    X_train, X_test, y_train, y_test = time_series_train_test_split(final_df)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    X_train.to_csv(str(OUTPUT_DIR / "X_train.csv"), index=False)
    X_test.to_csv(str(OUTPUT_DIR / "X_test.csv"), index=False)
    y_train.to_csv(str(OUTPUT_DIR / "y_train.csv"), index=False)
    y_test.to_csv(str(OUTPUT_DIR / "y_test.csv"), index=False)

    print(f"\nTrain and test sets saved in the {OUTPUT_DIR} directory.")
