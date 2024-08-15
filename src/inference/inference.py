import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Union

PROJECT_PATH = Path(os.getcwd())
sys.path.append(str(PROJECT_PATH))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils.logger import create_logger

logger = create_logger()


class ModelInference:
    """
    Class to load a model and standardization parameters and make predictions
    """

    def __init__(
        self,
        model: Union[RandomForestRegressor, XGBRegressor],
        standardization_params: dict,
    ):
        """
        Initializes the ModelInference class
        Args:
            model (Union[RandomForestRegressor, XGBRegressor]): The loaded model
            standardization_params (dict): The loaded standardization parameters
        Examples:
            >>> model = RandomForestRegressor()
            >>> standardization_params = {
            ...     "purchase_amount": {
            ...         "mean": 100,
            ...         "std": 10
            ...     }
            ... }
            >>> inference = ModelInference(model, standardization_params)
        """
        # Loads model and standardization parameters
        self.model = model
        self.standardization_params = standardization_params
        self.customer_data = pd.read_csv(
            PROJECT_PATH / "data" / "raw" / "customer_purchases.csv"
        )

    @staticmethod
    def load_model(model_path: Path) -> Union[RandomForestRegressor, XGBRegressor]:
        """
        Loads a model from a pickle file
        Args:
            model_path (Path): Path to the model pickle file
        Returns:
            Union[RandomForestRegressor, XGBRegressor]: The loaded model
        Examples:
            >>> model_path = Path("models/experiment_1/random_forest.pkl")
            >>> model = load_model(model_path)
        """
        # Loads model from pickle file
        with open(model_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_standardization_params(params_path: Path) -> dict:
        """
        Loads standardization parameters from a JSON file
        Args:
            params_path (Path): Path to the standardization parameters JSON file
        Returns:
            dict: The loaded standardization parameters
        Examples:
            >>> params_path = Path("models/experiment_1/random_forest_standardization_params.json")
            >>> standardization_params = load_standardization_params(params_path)
        """
        # Loads standardization parameters from JSON file
        with open(params_path, "r") as f:
            return json.load(f)

    def make_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the loaded model
        Args:
            data (pd.DataFrame): Preprocessed data for inference
        Returns:
            np.ndarray: Predictions
        Examples:
            >>> input_data = pd.DataFrame({
            ...     "customer_id": [1, 2],
            ...     "purchase_date": ["2021-01-01", "2021-01-02"],
            ...     "purchase_amount": [100, 200],
            ...     "annual_income": [50000, 60000],
            ...     "age": [20, 30],
            ...     "gender": ["Female", "Male"],
            ... })
            >>> inference = ModelInference(model, standardization_params)
        """
        # Makes predictions using the loaded model and given data
        predictions = self.model.predict(data)

        # Reverts standardization for the predictions
        reverted_predictions = (
            predictions * self.standardization_params["purchase_amount"]["std"]
            + self.standardization_params["purchase_amount"]["mean"]
        )
        return reverted_predictions


class EnsembleInference:
    """
    Class to make predictions using an ensemble of models
    """

    def __init__(
        self,
        models: List[Union[RandomForestRegressor, XGBRegressor]],
        standardization_params: List[dict],
    ):
        """
        Initializes the EnsembleInference class
        Args:
            model (Union[RandomForestRegressor, XGBRegressor]): The loaded model
            standardization_params (dict): The loaded standardization parameters
        Examples:
            >>> model = RandomForestRegressor()
            >>> standardization_params = {
            ...     "purchase_amount": {
            ...         "mean": 100,
            ...         "std": 10
            ...     }
            ... }
            >>> inference = EnsembleInference([model], [standardization_params])
        """
        # Initializes the models
        self.models = [
            ModelInference(model, params)
            for model, params in zip(models, standardization_params, strict=True)
        ]

    def make_predictions(self, preprocessed_data: list[pd.DataFrame]) -> np.ndarray:
        """
        Makes predictions using the ensemble of models
        Args:
            preprocessed_data (list[pd.DataFrame]): Preprocessed data for inference
        Returns:
            np.ndarray: Predictions
        Examples:
            >>> input_data = pd.DataFrame({
            ...     "customer_id": [1, 2],
            ...     "purchase_date": ["2021-01-01", "2021-01-02"],
            ...     "purchase_amount": [100, 200],
            ...     "annual_income": [50000, 60000],
            ...     "age": [20, 30],
            ...     "gender": ["Female", "Male"],
            ... })
            >>> ensemble = EnsembleInference([model_1, model_2], [standardization_params_1, standardization_params_2])
            >>> preprocessed_data = ensemble.preprocess_data(input_data)
            >>> predictions = ensemble.make_predictions(preprocessed_data)
        """
        # Makes predictions using all models and averages them
        predictions = [
            model.make_predictions(data)
            for model, data in zip(self.models, preprocessed_data, strict=True)
        ]

        # Averages the predictions from all models
        return np.mean(predictions, axis=0)


def run_inference(
    input_data: pd.DataFrame,
    models: List[Union[RandomForestRegressor, XGBRegressor]],
    standardization_params: List[dict],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Runs inference using an ensemble of models
    Args:
        input_data (pd.DataFrame): Input data for inference
        models (List[Union[RandomForestRegressor, XGBRegressor]]): List of loaded models
        standardization_params (List[dict]): List of loaded standardization parameters
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Tuple containing the input data with predictions and the predictions
    Examples:
        >>> input_data = pd.DataFrame({
            ...     "customer_id": [1, 2],
            ...     "purchase_date": ["2021-01-01", "2021-01-02"],
            ...     "purchase_amount": [100, 200],
            ...     "annual_income": [50000, 60000],
            ...     "age": [20, 30],
            ...     "gender": ["Female", "Male"],
            ... })
        >>> model = RandomForestRegressor()
        >>> standardization_params = {
            ...     "purchase_amount": {
            ...         "mean": 100,
            ...         "std": 10
            ...     }
            ... }
        >>> results, predictions = run_inference(input_data, [model], [standardization_params])
    """
    # Initializes the ensemble and makes predictions
    ensemble = EnsembleInference(models, standardization_params)
    predictions = ensemble.make_predictions([input_data])

    # Adds predictions to the original dataframe
    input_data["predicted_purchase_amount"] = predictions

    return input_data, predictions
