import json
import pickle
import sys
from pathlib import Path
from pprint import pprint
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pretty_errors  # noqa: F401
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

PROJECT_PATH = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_PATH))

from model_registry.model_registry import ModelInfo
from model_registry.utils import convert_numpy_types
from src.features.customer_features import add_customer_features
from src.features.numerical_features import (
    create_target_variable,
    fill_missing_monthly_purchases,
    standardize_features,
)
from src.features.time_features import add_time_features
from src.preprocessing.data_cleaner import clean_data
from src.preprocessing.data_loader import load_data
from src.training.time_series_split import time_series_train_test_split
from src.utils.logger import create_logger

logger = create_logger()


def load_and_preprocess_data(
    input_data_path: Path,
    output_data_path: Path = PROJECT_PATH
    / "data"
    / "processed"
    / "customer_purchases.csv",
) -> Tuple[pd.DataFrame, dict]:
    # Loads data from input_data_path
    df = load_data(str(input_data_path))

    # Preprocesses data
    cleaned_df = clean_data(df)
    time_featured_df = add_time_features(cleaned_df)
    customer_featured_df = add_customer_features(time_featured_df)
    missing_months_filled_df = fill_missing_monthly_purchases(customer_featured_df)
    standardized_df, standardization_params = standardize_features(
        missing_months_filled_df
    )
    final_df = create_target_variable(standardized_df)

    # Saves data and standardization params
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(str(output_data_path), index=False)

    return final_df, standardization_params


def train_random_forest(X_train, y_train):
    # Defines hyperparameters search space
    param_distributions = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": np.arange(3, 10),
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": np.arange(2, 10),
        "min_samples_leaf": np.arange(1, 10),
    }

    # Trains Random Forest model using RandomizedSearchCV
    model = RandomForestRegressor()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=5,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    return random_search


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    # Defines hyperparameters search space
    param_distributions = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": np.arange(3, 10),
        "learning_rate": np.linspace(0.01, 0.1, 10),
        "subsample": np.linspace(0.5, 1, 10),
        "colsample_bytree": np.linspace(0.5, 1, 10),
        "gamma": np.linspace(0, 1, 10),
        "reg_alpha": np.linspace(0, 1, 10),
        "reg_lambda": np.linspace(0, 1, 10),
    }

    xgb = XGBRegressor(enable_categorical=True)
    random_search = RandomizedSearchCV(
        xgb, param_distributions, n_iter=5, cv=5, n_jobs=-1, verbose=1
    )
    random_search.fit(X_train, y_train)
    return random_search


def evaluate_model(
    model: Union[RandomForestRegressor, XGBRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    standardization_params: dict,
) -> float:
    y_pred = model.predict(X_test)

    # Reverts standardization to get the real purchase amount
    reverted_y_test = (
        y_test * standardization_params["purchase_amount"]["std"]
        + standardization_params["purchase_amount"]["mean"]
    )
    reverted_y_pred = (
        y_pred * standardization_params["purchase_amount"]["std"]
        + standardization_params["purchase_amount"]["mean"]
    )

    # Calculates RMSE on the reverted values
    rmse = float(mean_squared_error(reverted_y_test, reverted_y_pred) ** 0.5)
    return rmse


def save_model(
    model: Union[RandomForestRegressor, XGBRegressor],
    output_model_path: Path | None = None,
) -> None:
    assert isinstance(
        model, (RandomForestRegressor, XGBRegressor)
    ), "model must be a RandomForestRegressor or XGBRegressor"
    assert output_model_path is not None, "output_model_path must be provided"

    # Saves the model to output_model_path
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)


def register_model(
    model_info: ModelInfo, model: Union[RandomForestRegressor, XGBRegressor]
) -> None:
    try:
        # Defines the URL for the model registry
        url = "http://localhost:8000/register_model"

        # Dumps model to bytes
        model_bytes = pickle.dumps(model)

        # Defines files to be sent in the request
        files = {"model_file": ("model.pkl", model_bytes)}

        # Prepares the data payload
        data = {
            "model_name": model_info.model_name,
            "version": model_info.version,
            "file_path": model_info.file_path,
            "standardization_params": json.dumps(model_info.standardization_params),
            "model_params": json.dumps(model_info.model_params),
            "additional_metadata": json.dumps(model_info.additional_metadata),
        }

        # Sends the request with individual fields
        response = requests.post(url, data=data, files=files)

        # Checks if the request was successful and logs the response
        if response.status_code in [200, 201]:
            logger.info(f"Model registered successfully: {response.json()['message']}")
        else:
            response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            logger.warning(f"Model registration failed: {e.response.json()['detail']}")
        elif e.response.status_code == 400:
            logger.error(f"Invalid model information: {e.response.json()['detail']}")
        else:
            logger.error(f"Error registering model: {e.response.json()['detail']}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to the model registry: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise


def main():
    INPUT_DATA_PATH = PROJECT_PATH / "data" / "raw" / "customer_purchases.csv"

    OUTPUT_DATA_PATH = PROJECT_PATH / "data" / "processed" / "customer_purchases.csv"

    # Loads and preprocesses data
    final_df, standardization_params = load_and_preprocess_data(
        INPUT_DATA_PATH,
        output_data_path=OUTPUT_DATA_PATH,
    )

    # Splits data into train and test sets using time_series_train_test_split
    X_train, X_test, y_train, y_test = time_series_train_test_split(final_df)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Trains and evaluates Random Forest
    logger.info("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    logger.info("Best Random Forest params:")
    pprint(rf_model.best_params_)
    rf_rmse = evaluate_model(
        rf_model.best_estimator_, X_test, y_test, standardization_params
    )
    logger.info(f"Random Forest RMSE: {rf_rmse}")
    try:
        rf_model_info = ModelInfo(
            model_name="random_forest",
            version="1.0",
            standardization_params=standardization_params,  # type: ignore
            model_params=convert_numpy_types(rf_model.best_estimator_.get_params()),  # type: ignore
            additional_metadata={
                "rmse": float(rf_rmse) if not np.isnan(rf_rmse) else None
            },
        )
        register_model(rf_model_info, rf_model.best_estimator_)
    except Exception as e:
        logger.error(f"Failed to register Random Forest model: {str(e)}")

    # Loads and preprocesses data
    final_df, standardization_params = load_and_preprocess_data(
        INPUT_DATA_PATH,
        output_data_path=OUTPUT_DATA_PATH,
    )

    # Splits data into train and test sets using time_series_train_test_split
    X_train, X_test, y_train, y_test = time_series_train_test_split(final_df)

    # Trains and evaluates XGBoost
    logger.info("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    logger.info("Best XGBoost params:")
    pprint(xgb_model.best_params_)
    xgb_rmse = evaluate_model(xgb_model, X_test, y_test, standardization_params)
    logger.info(f"XGBoost RMSE: {xgb_rmse}")
    try:
        xgb_model_info = ModelInfo(
            model_name="xgboost",
            version="1.0",
            standardization_params=standardization_params,
            model_params=convert_numpy_types(xgb_model.best_estimator_.get_params()),  # type: ignore
            additional_metadata={"rmse": float(xgb_rmse)},
        )
        register_model(xgb_model_info, xgb_model.best_estimator_)
    except Exception as e:
        logger.error(f"Failed to register Random Forest model: {str(e)}")


if __name__ == "__main__":
    main()
