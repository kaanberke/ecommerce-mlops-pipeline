# Data Processing and Feature Engineering

This section describes the data processing and feature engineering pipeline used in the project. These components are very important for preparing the data for model training and inference.

## Data Loading

The `src/preprocessing/data_loader.py` module handles loading data from various sources, including:

- CSV files
- Databases
- API endpoints

## Data Cleaning

The `src/preprocessing/data_cleaner.py` module is responsible for cleaning the raw data. Key operations include:

- Handling missing values
- Removing duplicates
- Formatting date fields
- Ensuring consistent data types

## Feature Engineering

Feature engineering is performed in several steps:

1. Time-based Features (`src/features/time_features.py`):
   - Extract year, month, day, and day of week from purchase dates
   - Calculate time-based statistics (e.g., days since last purchase)

2. Customer Features (`src/features/customer_features.py`):
   - Compute customer lifetime value (CLV)
   - Generate customer segmentation features
   - Calculate purchase frequency and recency

3. Numerical Features (`src/features/numerical_features.py`):
   - Standardize numerical features
   - Create interaction features (e.g., age_income_interaction)

## Usage in Training and Inference

The data processing and feature engineering pipeline is used in both the model training process (`train.py`) and the Preprocessing Service for real-time inference.

## Testing

The data processing and feature engineering components  includes a comprehensive test suite to ensure its functionality. These tests are located in the `tests/` directory at the project root.

To run the tests:

1. Ensure you're in the project root directory.
2. Run the following commands:

   ```
   pytest tests/test_data_cleaner.py
   pytest tests/test_customer_features.py
   pytest tests/test_numerical_features.py
   pytest tests/test_time_features.py
   ```

The test suite covers the following components:

1. Data Cleaning (`test_data_cleaner.py`):
   - Validates the data cleaning process
   - Checks for proper handling of missing values, duplicates, and data type conversions

2. Customer Features (`test_customer_features.py`):
   - Tests the generation of customer-specific features
   - Verifies the calculation of customer lifetime value (CLV)
   - Checks the creation of customer segmentation features

3. Numerical Features (`test_numerical_features.py`):
   - Ensures correct standardization of numerical features
   - Tests the handling of outliers
   - Verifies the creation of interaction features

4. Time Features (`test_time_features.py`):
   - Checks the extraction of time-based features from purchase dates
   - Validates the calculation of time-based statistics
