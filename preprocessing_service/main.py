import sys
import warnings
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_PATH))

from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.customer_features import add_customer_features
from src.features.numerical_features import standardize_features
from src.features.time_features import add_time_features
from src.preprocessing.data_cleaner import clean_data
from src.utils.logger import create_logger

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


class ProcessedData(BaseModel):
    features: Dict[str, float]


class StandardizationParams(BaseModel):
    params: Dict[str, Dict[str, float]] = {
        "purchase_amount": {"mean": 1, "std": 1},
        "annual_income": {"mean": 1, "std": 1},
        "clv": {"mean": 1, "std": 1},
        "age_income_interaction": {"mean": 1, "std": 1},
    }


@app.post("/preprocess", response_model=ProcessedData)
async def preprocess(raw_data: RawData, standardization_params: StandardizationParams):
    try:
        input_df = pd.DataFrame(raw_data.dict(), index=[0])
        input_df["purchase_date"] = pd.to_datetime(input_df["purchase_date"])

        customer_data = pd.read_csv(
            PROJECT_PATH / "data" / "raw" / "customer_purchases.csv"
        )
        customer_data["purchase_date"] = pd.to_datetime(customer_data["purchase_date"])

        customer_id = input_df["customer_id"].iloc[0]
        all_customer_data = customer_data[customer_data["customer_id"] == customer_id]

        combined_data = pd.concat([all_customer_data, input_df]).drop_duplicates(
            subset=["customer_id", "purchase_date"], keep="last"
        )

        cleaned_df = clean_data(combined_data, mode="test")

        time_featured_df = add_time_features(cleaned_df)

        customer_featured_df = add_customer_features(time_featured_df)

        standardized_df, _ = standardize_features(
            customer_featured_df,
            mode="test",
            standardization_params=standardization_params.params,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Converting to PeriodArray/Index representation will drop timezone information.",
            )
            input_date = input_df["purchase_date"].dt.to_period("M")
        filtered_df = standardized_df[standardized_df["year_month"].isin(input_date)]

        columns_to_drop = ["customer_id", "purchase_date", "year_month"]
        filtered_df = filtered_df.drop(
            columns=[col for col in columns_to_drop if col in filtered_df.columns]
        )

        processed_data = filtered_df.to_dict(orient="records")[0]
        return ProcessedData(features=processed_data)
    except Exception as E:
        logger.error(f"Error during preprocessing: {str(E)}")
        raise HTTPException(
            status_code=500, detail=f"Preprocessing error: {str(E)}"
        ) from E


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
