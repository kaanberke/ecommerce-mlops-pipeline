import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw data from CSV file
    Args:
        file_path (str): Path to the CSV file
    Returns:
        pd.DataFrame: Raw data
    """

    assert isinstance(file_path, str), "File path must be a string"
    assert file_path.endswith(".csv"), "File must be in CSV format"

    try:
        df = pd.read_csv(file_path)
        df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    except FileNotFoundError as E:
        raise FileNotFoundError(f"File not found at {file_path}") from E
    except KeyError as E:
        raise KeyError("File must contain 'purchase_date' column") from E

    return df


if __name__ == "__main__":
    from pathlib import Path

    DATA_PATH = str(
        Path(__file__).resolve().parents[2] / "data/raw/customer_purchases.csv"
    )

    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
