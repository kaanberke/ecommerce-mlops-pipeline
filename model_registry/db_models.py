from pathlib import Path

from sqlalchemy import JSON, Column, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class ModelInfoDB(Base):
    """
    Database model for storing model information
    """

    __tablename__ = "model_info"

    # Defines columns for the model_info table
    id = Column(Integer, primary_key=True, index=True)  # Primary key
    model_name = Column(String, nullable=False)  # Model's name
    version = Column(String, nullable=False)  # Model's version
    file_path = Column(String)  # Path to the model file
    standardization_params = Column(
        JSON
    )  # Standardization parameters used during training
    model_params = Column(JSON)  # Model parameters used during training
    additional_metadata = Column(JSON)  # Additional metadata

    # Defines a unique constraint for the model_name and version columns to prevent duplicates
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="_model_version_uc"),
    )


# Gets the directory of the current file
current_dir = Path(__file__).resolve().parent

# Creates a "data" directory inside the model_registry package
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)

# Sets the database file path
db_path = data_dir / "model_registry.db"

# Creates SQLite database engine using the db_path
engine = create_engine(f"sqlite:///{db_path}", echo=True)

# Creates all tables in the database
Base.metadata.create_all(bind=engine)

# Creates a SessionLocal class to manage the database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
