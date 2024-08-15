import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def clean_data(df: pd.DataFrame, mode: str = "train") -> pd.DataFrame:
    """
    Cleans the raw data by removing duplicates,
    handles missing values, and filters invalid entries

    Args:
        df (pd.DataFrame): Raw data
    Returns:
        pd.DataFrame: Cleaned data
    Examples:
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 1, 2, 2],
        ...     "purchase_date": ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"],
        ...     "purchase_amount": [100, 200, 300, 400],
        ...     "annual_income": [50000, 60000, 70000, 80000],
        ...     "age": [30, 40, 50, 60],
        ...     "gender": ["Male", "Female", "Male", "Female"],
        ...     "next_month_purchase_amount": [100, 200, 300, 400],
        ... })
        ... cleaned_df = clean_data(df)
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(mode, str), "Mode must be a string"
    assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], utc=True)

    # Drops the empty label column
    df = df.drop(columns=["next_month_purchase_amount"])

    # Sorts the dataframe by customer_id and purchase_date
    df = df.sort_values(["customer_id", "purchase_date"])

    if mode == "train":
        # Removes customers with none or only one purchase
        df = df.groupby("customer_id").filter(lambda x: len(x) > 1)

    # Removes negative purchase amounts and annual incomes, and non-positive ages
    df = df[(df["purchase_amount"] >= 0) & (df["annual_income"] >= 0) & (df["age"] > 0)]

    # Fills missing gender values with mode and converts to category
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
    df["gender"] = df["gender"].astype("category")

    # Removes remaining rows with missing values
    df = df.dropna()

    return df


if __name__ == "__main__":
    from pathlib import Path

    from data_loader import load_data

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )

    df = load_data(DATA_PATH)
    cleaned_df = clean_data(df)
    print(f"Data cleaned. Shape: {cleaned_df.shape}")
