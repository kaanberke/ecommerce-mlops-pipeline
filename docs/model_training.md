# Model Training

This document describes the model training process for predicting customer purchase amounts. The process is implemented in the `train.py` script, which handles data preprocessing, model training, evaluation, and registration.

## Overview

The training process consists of the following main steps:

1. Data loading and preprocessing
2. Time series train-test split
3. Training and evaluation of two models:
   - Random Forest Regressor
   - XGBoost Regressor
4. Model registration with the Model Registry Service

## Data Loading and Preprocessing

The `load_and_preprocess_data` function handles data preparation:

1. Loads raw data from `data/raw/customer_purchases.csv`
2. Cleans the data using `clean_data` function
3. Adds time-based features with `add_time_features`
4. Generates customer-specific features using `add_customer_features`
5. Fills missing monthly purchases with `fill_missing_monthly_purchases`
6. Standardizes features using `standardize_features`
7. Creates the target variable with `create_target_variable`
8. Saves the processed data to `data/processed/customer_purchases.csv`

## Time Series Train-Test Split

The `time_series_train_test_split` function is used to split the data into training and test sets, ensuring that the temporal data is respected.

## Training Procedure

Two models are trained using RandomizedSearchCV for hyperparameter tuning:

### Random Forest Regressor

- Hyperparameters tuned:
  - `n_estimators`: 100 to 1000
  - `max_depth`: 3 to 10
  - `max_features`: "auto", "sqrt", "log2"
  - `min_samples_split`: 2 to 10
  - `min_samples_leaf`: 1 to 10

### XGBoost Regressor

- Hyperparameters tuned:
  - `n_estimators`: 100 to 1000
  - `max_depth`: 3 to 10
  - `learning_rate`: 0.01 to 0.1
  - `subsample`: 0.5 to 1
  - `colsample_bytree`: 0.5 to 1
  - `gamma`: 0 to 1
  - `reg_alpha`: 0 to 1
  - `reg_lambda`: 0 to 1

Both models use 5-fold cross-validation and perform 5 iterations of random search. **While the number of iterations is set to 5 in this implementation, it can be adjusted based on computational resources and time constraints.**

## Model Evaluation

Models are evaluated using Root Mean Squared Error (RMSE) on the test set. The `evaluate_model` function:

1. Makes predictions on the test set
2. Reverts the standardization of both predictions and actual values
3. Calculates RMSE on the reverted values

## Model Registration

Trained models are registered with the Model Registry Service using the `register_model` function:

1. Serializes the model using pickle
2. Prepares model metadata, including:
   - Model name
   - Version
   - File path
   - Standardization parameters
   - Model parameters
   - Additional metadata (e.g., RMSE)
3. Sends a POST request to the Model Registry Service

## Usage

To run the training process:

1. Ensure all dependencies are installed
2. Make sure the raw data is available in `data/raw/customer_purchases.csv`
3. Run the script:

   ```
   python train.py
   ```

## Output

The script will:

1. Save the processed data to `data/processed/customer_purchases.csv`
2. Log information about the training process, including:
   - Data shapes
   - Best hyperparameters for each model
   - RMSE for each model
3. Register the trained models with the Model Registry Service

## Error Handling

The script includes error handling for:

- Model registration failures
- HTTP errors when communicating with the Model Registry Service
- Unexpected errors during the training process

Errors are logged using a custom logger created with `create_logger()`.

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- requests
