import pandas as pd


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds customer-specific features to the dataframe
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with customer features
    Examples:
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 1, 2, 2],
        ...     "purchase_amount": [100, 200, 300, 400],
        ...     "annual_income": [50000, 60000, 70000, 80000],
        ...     "age": [25, 30, 35, 40],
        ...     "year_month": ["2021-01", "2021-02", "2021-01", "2021-02"],
        ... })
        >>> add_customer_features(df)
           customer_id  purchase_amount  annual_income  age  year_month        clv age_group income_group  age_income_interaction
        0            1              100          50000   25     2021-01      100.0     19-25     Very Low                1250000
        1            1              200          60000   30     2021-02      300.0     26-35          Low                1800000
        2            2              300          70000   35     2021-01      300.0     26-35          Low                2450000
        3            2              400          80000   40     2021-02      700.0     36-45       Medium                3200000

    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

    df = (
        df.groupby(["customer_id", "year_month"])
        .agg(
            {
                "age": lambda x: pd.Series.mode(x)[0],
                "gender": lambda x: pd.Series.mode(x)[0],
                "annual_income": "max",
                "purchase_amount": "sum",
                "frequency": "first",
                "days_since_last_purchase": "first",
                "avg_days_between_purchases": "first",
            }
        )
        .reset_index()
    )

    # Adds Customer Lifetime Value (CLV) column by cumulatively summing purchase_amount
    df["clv"] = df.groupby("customer_id")["purchase_amount"].cumsum()

    # Adds age group column based on predefined bins and labels
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 25, 35, 45, 55, 65, 100],
        labels=["0-18", "19-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )

    # One hot encodes age group column
    df = pd.get_dummies(df, columns=["age_group"], drop_first=True)

    # Adds income group column based on quantiles and labels
    # Use 'drop' for duplicates to handle duplicate bin edges
    df["income_group"] = pd.cut(
        df["annual_income"],
        bins=[0.0, 54415.0, 75395.0, 94999.0, 114295.0, 1015980],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    # One hot encodes income group column
    df = pd.get_dummies(df, columns=["income_group"], drop_first=True)

    # Adds age-income interaction column by multiplying age and annual_income
    df["age_income_interaction"] = df["age"] * df["annual_income"]

    # Changes gender to a binary column
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).astype(int)

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.features.time_features import add_time_features
    from src.preprocessing.data_cleaner import clean_data
    from src.preprocessing.data_loader import load_data

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )

    df = load_data(DATA_PATH)
    cleaned_df = clean_data(df)
    time_featured_df = add_time_features(cleaned_df)
    customer_featured_df = add_customer_features(time_featured_df)
    print(f"Customer features added. Shape: {customer_featured_df.shape}")
