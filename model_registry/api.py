import json
import logging
from io import BytesIO
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from sqlalchemy.orm import Session

from .db_models import get_db
from .model_registry import ModelInfo, registry

app = FastAPI()
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


@app.post("/register_model")
async def register_model(
    model_name: str = Form(...),
    version: str = Form(...),
    standardization_params: str = Form(...),
    model_params: str = Form(...),
    additional_metadata: str = Form("{}"),
    model_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        standardization_params_dict = json.loads(standardization_params)
        model_params_dict = json.loads(model_params)
        additional_metadata_dict = json.loads(additional_metadata)

        # Creates ModelInfo object
        model_info = ModelInfo(
            model_name=model_name,
            version=version,
            file_path="",
            standardization_params=standardization_params_dict,
            model_params=model_params_dict,
            additional_metadata=additional_metadata_dict,
        )

        # Reads the uploaded file contents
        contents = await model_file.read()

        # Deserializes the model from the file contents
        model = joblib.load(BytesIO(contents))

        # Saves the model to a file using joblib
        model_path = str(
            MODEL_DIR / f"{model_info.model_name}_{model_info.version}.joblib"
        )
        joblib.dump(model, model_path)

        # Updates model_info with the file path
        model_info.file_path = model_path

        # Registers the model info in the database
        registry.register_model(model_info)

        return {
            "message": f"Model '{model_info.model_name}' version '{model_info.version}' registered successfully",
            "file_path": model_path,
        }
    except Exception as E:
        logger.error(f"Error registering model: {str(E)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(E)}"
        ) from E


@app.delete("/remove_model/{model_name}/{version}")
def remove_model(model_name: str, version: str, db: Session = Depends(get_db)):
    """
    Removes a model from the model registry
    Parameters
    ----------
    model_name : str
        Name of the model
    version : str
        Version of the model
    Returns
    -------
    dict
        Dictionary containing the success message
    """
    try:
        # Gets the model info before removing it
        model_info = registry.get_model_info(model_name, version)

        # Removes the model from the registry
        registry.remove_model(model_name, version)

        # If the model has a file associated with it, removes the file
        if model_info.file_path and Path(model_info.file_path).exists():
            Path(model_info.file_path).unlink()

        return {
            "message": f"Model '{model_name}' version '{version}' removed successfully"
        }
    except ValueError as E:
        raise HTTPException(status_code=404, detail=str(E)) from E
    except Exception as E:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while removing the model: {str(E)}",
        ) from E


@app.get("/get_model_info/{model_name}/{version}")
def get_model_info(model_name: str, version: str, db: Session = Depends(get_db)):
    """
    Retrieves model information from the model registry
    Parameters
    ----------
    model_name : str
        Name of the model
    version : str
        Version of the model
    Returns
    -------
    ModelInfo
        Model information
    """
    try:
        return registry.get_model_info(model_name, version)
    except ValueError as E:
        raise HTTPException(status_code=404, detail=str(E)) from E


@app.get("/get_latest_model_info/{model_name}")
def get_latest_model_info(model_name: str, db: Session = Depends(get_db)):
    """
    Retrieves the latest model information from the model registry
    Parameters
    ----------
    model_name : str
        Name of the model
    Returns
    -------
    ModelInfo
        Model information
    """
    try:
        return registry.get_latest_model_info(model_name)
    except ValueError as E:
        raise HTTPException(status_code=404, detail=str(E)) from E


@app.get("/list_models")
def list_models(db: Session = Depends(get_db)):
    """
    Lists all models in the model registry
    Returns
    -------
    List[ModelInfo]
        List of model information
    """
    return registry.list_models()


@app.get("/get_model/{model_name}/{version}")
def get_model(model_name: str, version: str, db: Session = Depends(get_db)):
    """
    Retrieves the model file from the model registry
    Parameters
    ----------
    model_name : str
        Name of the model
    version : str
        Version of the model
    Returns
    -------
    Response
        File response containing the model file
    """
    try:
        model_info = registry.get_model_info(model_name, version)
        if not model_info or not model_info.file_path:
            raise HTTPException(status_code=404, detail="Model file path not found")

        with open(model_info.file_path, "rb") as f:
            model_bytes = f.read()

        return Response(content=model_bytes, media_type="application/octet-stream")
    except ValueError as E:
        raise HTTPException(status_code=404, detail=str(E)) from E
    except Exception as E:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading the model: {str(E)}",
        ) from E


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
