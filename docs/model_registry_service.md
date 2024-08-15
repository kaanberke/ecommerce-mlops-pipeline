# Model Registry Service

The Model Registry Service is responsible for storing, versioning, and retrieving machine learning models. It provides a centralized repository for managing model artifacts and metadata.

## API Documentation

FastAPI automatically generates interactive API documentation (Swagger UI) for this service. To access it:

1. Start the Model Registry Service.
2. Open a web browser and navigate to `http://localhost:8000/docs`.

This interactive documentation allows you to:

- View all available endpoints
- See request and response schemas
- Test the API directly from the browser

Additionally, a ReDoc version of the documentation is available at `http://localhost:8000/redoc`.

## Key Components

1. `api.py`: Defines the FastAPI endpoints for the service.
2. `model_registry.py`: Implements the core functionality of the model registry.
3. `db_models.py`: Defines the database schema for storing model metadata.
4. `utils.py`: Contains utility functions for data type conversion and cleaning.

## API Endpoints

- `POST /register_model`: Register a new model with metadata.
- `GET /get_model_info/{model_name}/{version}`: Retrieve model information.
- `GET /get_latest_model_info/{model_name}`: Get the latest version of a model.
- `GET /list_models`: List all registered models and versions.
- `GET /get_model/{model_name}/{version}`: Download a specific model file.
- `DELETE /remove_model/{model_name}/{version}`: Remove a specific model version.

## Database Schema

The `ModelInfoDB` class in `db_models.py` defines the schema for storing model metadata:

- `id`: Primary key
- `model_name`: Name of the model
- `version`: Version of the model
- `file_path`: Path to the model file
- `standardization_params`: Standardization parameters used during training
- `model_params`: Model parameters used during training
- `additional_metadata`: Any additional metadata

## Usage

To run the Model Registry Service:

1. Ensure the required dependencies are installed.
2. Run `uvicorn model_registry.api:app --reload` to start the service.
3. Access the API documentation at `http://localhost:8000/docs`.

## Testing

The Model Registry Service includes a comprehensive test suite to ensure its functionality. These tests are located in the `tests/` directory at the project root.

To run the tests:

1. Ensure you're in the project root directory.
2. Run the following command:

   ```
   pytest tests/test_model_registry.py
   ```

The test suite includes:

- `test_model_registry.py`: Contains unit tests for the ModelRegistry class, covering:
  - Registering a model
  - Retrieving model information
  - Getting the latest model version
  - Listing all models
  - Handling duplicate model registration
  - Handling requests for non-existent models

Key test cases:

- `test_register_model`: Verifies that a model can be registered successfully
- `test_get_model_info`: Checks if registered model information can be retrieved correctly
- `test_get_latest_model_info`: Ensures that the latest version of a model can be retrieved
- `test_list_models`: Validates the listing of all registered models
- `test_register_duplicate_model`: Checks that registering a duplicate model raises an error
- `test_get_nonexistent_model`: Verifies that requesting a non-existent model raises an error

The tests use pytest fixtures defined in `tests/conftest.py` to set up a test database session and sample model information.
