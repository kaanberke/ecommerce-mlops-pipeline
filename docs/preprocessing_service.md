# Preprocessing Service

The Preprocessing Service is responsible for cleaning raw data and generating features for model training and inference. It ensures that incoming data is properly formatted and contains all necessary features for making predictions.

## API Documentation

FastAPI automatically generates interactive API documentation (Swagger UI) for this service. To access it:

1. Start the Preprocessing Service.
2. Open a web browser and navigate to `http://localhost:8001/docs`.

This interactive documentation allows you to:

- View all available endpoints
- See request and response schemas
- Test the API directly from the browser

Additionally, a ReDoc version of the documentation is available at `http://localhost:8001/redoc`.

## Key Components

1. `main.py`: Defines the FastAPI application and endpoints for the preprocessing service.
2. `src/features/`: Contains modules for feature engineering:
   - `customer_features.py`: Generates customer-specific features.
   - `numerical_features.py`: Processes numerical features and standardizes them.
   - `time_features.py`: Creates time-based features.
3. `src/preprocessing/data_cleaner.py`: Implements data cleaning functionality.

## API Endpoint

- `POST /preprocess`: Processes raw data and returns features ready for model inference.

## Input Data Format

The service expects input data in the following format:

```python
class RawData(BaseModel):
    customer_id: int
    purchase_date: str
    purchase_amount: float
    annual_income: float
    age: int
    gender: str
```

## Preprocessing Steps

1. Convert input data to a pandas DataFrame.
2. Load historical customer data.
3. Clean the data.
4. Add time-based features.
5. Add customer-specific features.
6. Standardize numerical features.
7. Filter and prepare the final feature set for the input data.

## Usage

To run the Preprocessing Service:

1. Ensure the required dependencies are installed.
2. Run `uvicorn preprocessing_service.main:app --reload` to start the service.
3. Send POST requests to `http://localhost:8001/preprocess` with the appropriate JSON payload.

For example, you can use `curl` to send a POST request:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8001/preprocess' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "raw_data": {
    "customer_id": 1,
    "purchase_date": "2023-01-01T00:00:00",
    "purchase_amount": 50,
    "annual_income": 50000,
    "age": 30,
    "gender": "Female"
  },
  "standardization_params": {
    "params": {
      "purchase_amount": {
        "mean": 1,
        "std": 1
      },
      "annual_income": {
        "mean": 1,
        "std": 1
      },
      "clv": {
        "mean": 1,
        "std": 1
      },
      "age_income_interaction": {
        "mean": 1,
        "std": 1
      }
    }
  }
}'
```
