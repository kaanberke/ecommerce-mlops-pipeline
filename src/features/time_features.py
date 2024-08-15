import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features to the dataframe

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with time features
    Examples:
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 1, 2, 2],
        ...     "purchase_date": ["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"],
        ...     "purchase_amount": [100, 200, 300, 400],
        ...     "annual_income": [50000, 60000, 70000, 80000],
        ...     "age": [30, 40, 50, 60],
        ... })
        ... time_featured_df = add_time_features(df)
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    # Removes timezone information if exists
    if df["purchase_date"].dt.tz is not None:
        df["purchase_date"] = df["purchase_date"].dt.tz_localize(None)

    # Converts purchase_date to year_month, year, month, and quarter
    df["year_month"] = df["purchase_date"].dt.to_period("M")
    df["year"] = df["purchase_date"].dt.year.astype("int")
    df["month"] = df["purchase_date"].dt.month.astype("int")
    df["quarter"] = df["purchase_date"].dt.quarter.astype("int")

    # Adds frequency column
    df["frequency"] = df.groupby(["customer_id", "year_month"])[
        "customer_id"
    ].transform("count")

    # Adds days since last purchase
    df["days_since_last_purchase"] = (
        df.groupby("customer_id")["purchase_date"].diff().dt.days.fillna(-1)
    )

    # Calculates average time between purchases
    df["avg_days_between_purchases"] = (
        df.groupby("customer_id")["days_since_last_purchase"]
        .transform("mean")
        .astype(int)
    )

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.preprocessing.data_cleaner import clean_data
    from src.preprocessing.data_loader import load_data

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )

    df = load_data(DATA_PATH)
    cleaned_df = clean_data(df)
    time_featured_df = add_time_features(cleaned_df)
    print(f"Time features added. Shape: {time_featured_df.shape}")
