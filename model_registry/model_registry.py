from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from .db_models import ModelInfoDB, get_db
from .utils import convert_numpy_types, remove_none_values


class ModelInfo(BaseModel):
    """
    Model information class used to organize and store model information
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str  # Model's name
    version: str  # Model's version
    file_path: Optional[str] = None  # Path to the model file
    standardization_params: Dict[
        str, Any
    ]  # Standardization parameters used during training
    model_params: Dict[str, Any]  # Model parameters used during training
    additional_metadata: Dict[str, Any] = Field(
        default_factory=dict
    )  # Additional metadata

    def dict(self, *args, **kwargs):  # type: ignore
        """
        Converts the model information to a dictionary
        by converting numpy types to native python types
        and removing None values
        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        Returns:
            dictionary representation of the model information
        """
        d = super().dict(*args, **kwargs)
        converted = convert_numpy_types(d)
        return remove_none_values(converted)


class ModelRegistry:
    """
    Model registry class used to register, retrieve, and list
    model information
    """

    def __init__(self):
        self.db = next(get_db())

    def register_model(self, model_info: ModelInfo):
        """
        Registers a model in the model registry
        Args:
            model_info: Model information to register
        Returns:
            Model information
        Raises:
            ValueError: if the model already exists in the registry
            or if there is an error registering the model
        """
        try:
            # Converts model info to DB model info
            db_model_info = ModelInfoDB(**model_info.dict())
            # Adds model info to the database
            self.db.add(db_model_info)
            # Commits the transaction
            self.db.commit()
            # Refreshes the model info
            self.db.refresh(db_model_info)

            return db_model_info
        except IntegrityError as E:
            self.db.rollback()
            raise ValueError(
                f"Model '{model_info.model_name}' version '{model_info.version}' already exists in the registry."
            ) from E
        except Exception as E:
            self.db.rollback()
            raise ValueError(f"Error registering model: {str(E)}") from E

    def remove_model(self, model_name: str, version: str):
        """
        Removes a model from the model registry
        Args:
            model_name: Model's name
            version: Model's version
        Raises:
            ValueError: if the model does not exist in the registry
        """
        try:
            db_model_info = (
                self.db.query(ModelInfoDB)
                .filter(
                    ModelInfoDB.model_name == model_name, ModelInfoDB.version == version
                )
                .first()
            )
            if not db_model_info:
                raise ValueError(
                    f"Model {model_name} version {version} not found in registry"
                )

            self.db.delete(db_model_info)
            self.db.commit()
        except Exception as E:
            self.db.rollback()
            raise ValueError(f"Error removing model: {str(E)}") from E

    def get_model_info(self, model_name: str, version: str) -> ModelInfo:
        """
        Retrieves model information from the model registry
        Args:
            model_name: Model's name
            version: Model's version
        Returns:
            Model information
        Raises:
            ValueError: if the model does not exist in the
            registry
        """
        # Retrieves model info from the database using the model name and version
        db_model_info = (
            self.db.query(ModelInfoDB)
            .filter(
                ModelInfoDB.model_name == model_name, ModelInfoDB.version == version
            )
            .first()
        )
        if not db_model_info:
            raise ValueError(
                f"Model {model_name} version {version} not found in registry"
            )
        return ModelInfo.parse_obj(db_model_info.__dict__)

    def get_latest_model_info(self, model_name: str) -> ModelInfo:
        """
        Retrieves the latest model information from the model registry
        Args:
            model_name: Model's name
        Returns:
            Model information
        Raises:
            ValueError: if the model does not exist in the
            registry
        """
        db_model_info = (
            self.db.query(ModelInfoDB)
            .filter(ModelInfoDB.model_name == model_name)
            .order_by(ModelInfoDB.version.desc())
            .first()
        )
        if not db_model_info:
            raise ValueError(f"Model {model_name} not found in registry")
        return ModelInfo.parse_obj(db_model_info.__dict__)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists all models in the model registry
        Returns:
            List of dictionaries containing model name and versions
        """
        models = self.db.query(ModelInfoDB.model_name, ModelInfoDB.version).all()
        model_dict = {}
        for model in models:
            if model.model_name not in model_dict:
                model_dict[model.model_name] = []
            model_dict[model.model_name].append(model.version)
        return [{"model_name": k, "versions": v} for k, v in model_dict.items()]


# Initializes the model registry
registry = ModelRegistry()
