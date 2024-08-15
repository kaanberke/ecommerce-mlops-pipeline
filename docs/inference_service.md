# Inference Service

The Inference Service is responsible for making predictions using the trained machine learning models. It integrates with the Model Registry Service to fetch the requested model's version and uses the Preprocessing Service to prepare input data.

## API Documentation

FastAPI automatically generates interactive API documentation (Swagger UI) for this service. To access it:

1. Start the Inference Service.
2. Open a web browser and navigate to `http://localhost:8002/docs`.

This interactive documentation allows you to:

- View all available endpoints
- See request and response schemas
- Test the API directly from the browser

Additionally, a ReDoc version of the documentation is available at `http://localhost:8002/redoc`.

## Key Components

1. `main.py`: Defines the FastAPI application and endpoints for the inference service.
2. `config.py`: Contains configuration settings for the service.
3. `utils.py`: Provides utility functions for the inference service.
4. `src/inference/inference.py`: Implements the core inference logic.

## API Endpoint

- `POST /predict`: Accepts raw customer data and returns a prediction for the next month's purchase amount.

## Prediction Process

1. Receive raw customer data.
2. Preprocess the data using the Preprocessing Service.
3. Load the latest model from the Model Registry Service.
4. Run inference using the preprocessed data and loaded model.
5. Return the prediction result.

## Input Data Format

The service expects input data in the same format as the Preprocessing Service:

```python
class RawData(BaseModel):
    customer_id: int
    purchase_date: str
    purchase_amount: float
    annual_income: float
    age: int
    gender: str
```

## Usage

To run the Inference Service:

1. Ensure the required dependencies are installed.
2. Make sure the Model Registry Service and Preprocessing Service are running.
3. Run `uvicorn inference_service.main:app --reload` to start the service.
4. Send POST requests to `http://localhost:8002/predict` with the appropriate JSON payload.

For example, you can use `curl` to send a POST request:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8002/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "random_forest",
  "model_version": "1.0",
  "purchase_data": {
    "customer_id": 1,
    "purchase_date": "2023-01-01T00:00:00",
    "purchase_amount": 50,
    "annual_income": 50000,
    "age": 30,
    "gender": "Female"
  }
}'
```


## Testing

The Inference Service includes both unit and integration tests, located in the `inference_service/tests/` directory.

To run the tests:

1. Make sure you're in the project root directory.
2. Run the following commands:

   For unit tests:
   ```
   pytest inference_service/tests/test_inference_service.py
   ```

   For integration tests:
   ```
   pytest inference_service/tests/test_inference_service_integration.py
   ```

The test suite includes:

1. `test_inference_service.py`: Unit tests for individual components of the inference service
   - `test_predict_endpoint`: Validates the prediction endpoint's behavior
   - `test_list_available_models`: Checks the listing of available models
   - `test_load_model`: Ensures correct loading of models from the Model Registry

2. `test_inference_service_integration.py`: Integration tests that cover the entire prediction pipeline
   - `test_preprocessing`: Verifies the preprocessing step
   - `test_end_to_end_prediction_with_preprocessing`: Tests the complete prediction workflow, including model registration, preprocessing, and prediction
   - `test_list_models_integration`: Checks the integration with the Model Registry for listing models

These tests verify:

- Correct loading of models from the Model Registry
- Proper preprocessing of input data
- Error handling and edge cases
- Integration between different components of the system

The tests use mocking to isolate components and control external dependencies. They also interact with the Model Registry and Preprocessing services to ensure end-to-end functionality.
