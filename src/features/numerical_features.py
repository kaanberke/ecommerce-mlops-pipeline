import json

import numpy as np
import pandas as pd


def standardize_column(
    df: pd.DataFrame,
    col: str,
    params: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Standardizes a column, only considering positive values
    Args:
        df (pd.DataFrame): Input dataframe
        col (str): Column to standardize
        params (dict[str, float] | None): Pre-computed standardization parameters
    Returns:
        tuple[np.ndarray, dict[str, float]]: Standardized column and standardization parameters
    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        >>> standardized, params = standardize_column(df, "A")
        >>> standardized
        array([-1.26491106, -0.63245553,  0.        ,  0.63245553,  1.26491106])
        >>> params
        {'mean': 3.0, 'std': 1.5811388300841898}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not isinstance(col, str):
        raise TypeError("Column must be a string")
    if col not in df.columns:
        raise ValueError(f"{col} not in dataframe columns")

    column_data = df[col].to_numpy()
    positive_mask = column_data > 0

    if params is None:
        positive_data = column_data[positive_mask]
        mean = np.mean(positive_data)
        std = np.std(positive_data)
        params = {"mean": mean, "std": std}  # type: ignore
    else:
        mean = params["mean"]
        std = params["std"]

    # Standardizes column using mean and standard deviation
    standardized = np.zeros_like(column_data).astype(float)
    standardized[positive_mask] = np.divide(column_data[positive_mask] - mean, std)

    return standardized, params  # type: ignore


def standardize_features(
    df: pd.DataFrame,
    mode: str = "train",
    standardization_params: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Standardizes numerical features in the dataframe
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        tuple[pd.DataFrame, dict]: Dataframe with standardized features and standardization parameters
    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "purchase_amount": [100, 200, 300, 400, 500],
        ...         "annual_income": [50000, 60000, 70000, 80000, 90000],
        ...         "clv": [100, 300, 600, 1000, 1500],
        ...         "age_income_interaction": [50000, 120000, 210000, 320000, 450000],
        ...     }
        ... )
        >>> standardized_df, params = standardize_features(df)
        >>> standardized_df
            purchase_amount_standardized  annual_income_standardized  clv_standardized  age_income_interaction_standardized
        0                      -1.264911                   -1.264911         -1.264911                            -1.264911
        1                      -0.632456                   -0.632456         -0.632456                            -0.632456
        2                       0.000000                    0.000000          0.000000                             0.000000
        3                       0.632456                    0.632456          0.632456                             0.632456
        4                       1.264911                    1.264911          1.264911                             1.264911
        >>> params
        {'purchase_amount': {'mean': 300.0, 'std': 158.11388300841898}, 'annual_income': {...}, ...}
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(mode, str), "Mode must be a string"
    assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
    # If mode is test, standardization params must be provided
    if mode == "test":
        assert isinstance(
            standardization_params, dict
        ), "Standardization params must be a dictionary"

    columns_to_standardize = [
        "purchase_amount",
        "annual_income",
        "clv",
        "age_income_interaction",
    ]

    if mode == "train":
        standardization_params = {}

        # Standardizes each column and drops original column
        for col in columns_to_standardize:
            df[f"{col}_standardized"], params = standardize_column(df, col)
            standardization_params[col] = params
            df.drop(columns=[col], inplace=True)

        return df, standardization_params

    elif mode == "test":
        for col in columns_to_standardize:
            standardized, _ = standardize_column(df, col, standardization_params[col])
            df[f"{col}_standardized"] = standardized
            df.drop(columns=[col], inplace=True)

        return df, standardization_params


def fill_missing_monthly_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing monthly purchases with 0 and forward fills other missing values
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with filled missing values
    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "customer_id": [1, 1],
        ...         "year_month": ["2020-01", "2020-03"],
        ...         "age": [25, 25],
        ...         "gender": ["Male", "Male"],
        ...         "annual_income": [50000, 50000],
        ...         "purchase_amount": [100, 200],
        ...         "frequency": [1, 1,],
        ...         "days_since_last_purchase": [-1, 20],
        ...         "avg_days_between_purchases": [20, 20],
        ...     }
        ... )
        >>> fill_missing_monthly_purchases(df)
              customer_id year_month  age gender  annual_income  purchase_amount  frequency  days_since_last_purchase  avg_days_between_purchases
        0            1    2020-01      25   Male          50000            100.0        1.0                      -1.0                        20.0
        1            1    2020-02      25   Male          50000              0.0        0.0                      30.0                        20.0
        2            1    2020-03      25   Male          50000            200.0        1.0                      20.0                        20.0
    """
    all_data = []

    for customer_id, group in df.groupby("customer_id"):
        start_date = group["year_month"].min()
        end_date = group["year_month"].max()

        all_months = pd.period_range(start=start_date, end=end_date, freq="M")

        complete_group = pd.DataFrame(
            {"customer_id": customer_id, "year_month": all_months}
        )

        complete_group = complete_group.merge(
            group, on=["customer_id", "year_month"], how="left"
        )

        # Forward fill other missing values
        complete_group["age"] = complete_group["age"].fillna(method="ffill")  # type: ignore
        complete_group["gender"] = complete_group["gender"].fillna(method="ffill")  # type: ignore
        complete_group["annual_income"] = complete_group["annual_income"].fillna(
            method="ffill"
        )  # type: ignore
        complete_group["avg_days_between_purchases"] = complete_group[
            "avg_days_between_purchases"
        ].fillna(method="ffill")  # type: ignore

        # Fills missing purchase_amount with 0 to represent inactivity
        complete_group["purchase_amount"] = complete_group["purchase_amount"].fillna(0)

        # Fills missing frequency with 0 to represent inactivity
        complete_group["frequency"] = complete_group["frequency"].fillna(0)

        # Fills missing days_since_last_purchase with 0 to represent inactivity
        complete_group["days_since_last_purchase"] = complete_group[
            "days_since_last_purchase"
        ].fillna(0)

        for i in range(1, len(complete_group)):
            if complete_group.iloc[i]["purchase_amount"] == 0:
                complete_group.at[i, "days_since_last_purchase"] = (
                    complete_group.iloc[i - 1]["days_since_last_purchase"] + 30
                )

        all_data.append(complete_group)

    df = pd.concat(all_data).reset_index(drop=True)

    # Changes all object columns to category
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    return df


def create_target_variable(df: pd.DataFrame):
    """
    Creates the target variable for next month's purchase amount
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with target variable
    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "customer_id": [1, 1, 1, 1, 1],
        ...         "purchase_amount_standardized": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> create_target_variable(df)
           customer_id  purchase_amount_standardized  next_month_purchase_amount
        0            1                            1                          2.0
        1            1                            2                          3.0
        2            1                            3                          4.0
        3            1                            4                          5.0
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

    # Shifts purchase_amount_standardized by 1 to get next month's purchase amount
    df["next_month_purchase_amount"] = df.groupby("customer_id")[
        "purchase_amount_standardized"
    ].shift(-1)
    df = df.dropna(subset=["next_month_purchase_amount"])

    return df


def save_standardization_params(params: dict, file_path: str):
    """
    Saves standardization parameters to a JSON file
    Args:
        params (dict): Standardization parameters
        file_path (str): Path to save the JSON file
    Returns:
        None
    Examples:
        >>> params = {'purchase_amount': {'mean': 300.0, 'std': 158.11388300841898}, ...}
        >>> save_standardization_params(params, "standardization_params.json")
    """

    assert isinstance(params, dict), "Input must be a dictionary"
    assert isinstance(file_path, str), "File path must be a string"
    assert file_path.endswith(".json"), "File path must be a JSON file"

    with open(file_path, "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.features.customer_features import add_customer_features
    from src.features.time_features import add_time_features
    from src.preprocessing.data_cleaner import clean_data
    from src.preprocessing.data_loader import load_data

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )
    OUTPUT_PATH = str(
        Path(__file__).resolve().parents[2]
        / "data/processed/preprocessed_customer_purchases.csv"
    )

    df = load_data(DATA_PATH)
    cleaned_df = clean_data(df)
    time_featured_df = add_time_features(cleaned_df)
    customer_featured_df = add_customer_features(time_featured_df)
    standardized_df, standardization_params = standardize_features(customer_featured_df)
    final_df = create_target_variable(standardized_df)

    print(f"Numerical features processed. Final shape: {final_df.shape}")

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")

    PARAMS_PATH = str(
        Path(__file__).resolve().parents[1] / "models/standardization_params.json"
    )

    save_standardization_params(standardization_params, PARAMS_PATH)
    print(f"Standardization parameters saved to {PARAMS_PATH}")
