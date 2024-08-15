import sys
from pathlib import Path

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Dict, List

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.inference import run_inference
from src.utils.logger import create_logger

from .config import PREPROCESSING_SERVICE_URL
from .utils import list_available_models, load_model

app = FastAPI()
logger = create_logger()


class RawData(BaseModel):
    customer_id: int = Field(
        ..., examples=[1], description="Unique identifier for the customer"
    )
    purchase_date: str = Field(
        ...,
        examples=["2023-01-01T00:00:00"],
        description="Date and time of the purchase",
    )
    purchase_amount: float = Field(
        ..., examples=[50.0], description="Amount of the purchase"
    )
    annual_income: float = Field(
        ..., examples=[50000.0], description="Annual income of the customer"
    )
    age: int = Field(..., examples=[30], description="Age of the customer")
    gender: str = Field(..., examples=["Female"], description="Gender of the customer")


class StandardizationParams(BaseModel):
    params: Dict[str, Dict[str, float]]


class PreprocessRequest(BaseModel):
    raw_data: RawData
    standardization_params: StandardizationParams


class PredictionRequest(BaseModel):
    model_name: str = "random_forest"
    model_version: str = "1.0"
    purchase_data: RawData


class PredictionResponse(BaseModel):
    customer_id: int
    purchase_date: str
    predicted_purchase_amount: float


async def preprocess_data(
    raw_data: RawData, standardization_params: StandardizationParams
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess",
                json={
                    "raw_data": raw_data.dict(),
                    "standardization_params": standardization_params.dict(),
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as E:
        logger.error(f"HTTP error occurred while preprocessing data: {E}")
        raise HTTPException(
            status_code=500, detail=f"Preprocessing service error: {str(E)}"
        ) from E
    except Exception as E:
        logger.error(f"Unexpected error occurred while preprocessing data: {E}")
        raise HTTPException(
            status_code=500, detail="Unexpected error during preprocessing"
        ) from E


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest = Body(...),
):
    """
    Predicts the purchase amount for a customer based on the input data
    """
    try:
        # Loads the specified model and standardization parameters
        model, standardization_params = load_model(
            request.model_name, request.model_version
        )
        if model is None or standardization_params is None:
            raise ValueError(
                f"Failed to load model {request.model_name} version {request.model_version}"
            )

        # Preprocesses the data
        processed_data = await preprocess_data(
            request.purchase_data, StandardizationParams(params=standardization_params)
        )

        # Runs inference
        processed_data_df = pd.DataFrame(processed_data["features"], index=[0])
        _, predictions = run_inference(
            processed_data_df, [model], [standardization_params]
        )

        # Prepares response
        response = PredictionResponse(
            customer_id=request.purchase_data.customer_id,
            purchase_date=request.purchase_data.purchase_date,
            predicted_purchase_amount=float(predictions[0]),
        )

        return response
    except ValueError as E:
        logger.error(f"Value error in predict endpoint: {str(E)}")
        raise HTTPException(status_code=400, detail=str(E)) from E
    except Exception as E:
        logger.error(f"Unexpected error in predict endpoint: {str(E)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(E)}"
        ) from E


@app.get("/models", response_model=List[dict])
async def get_available_models():
    """
    Retrieves the list of available models from the model registry
    Returns
    -------
    List[dict]
        List of available models
    """
    try:
        models = list_available_models()
        return models
    except Exception as E:
        logger.error(f"Error retrieving available models: {str(E)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(E)}"
        ) from E


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
