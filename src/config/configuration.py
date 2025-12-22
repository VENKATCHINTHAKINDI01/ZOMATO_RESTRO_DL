import os
from dotenv import load_dotenv
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from src.constant.application import ARTIFACTS_DIR

load_dotenv()

class ConfigurationManager:

    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("DATABASE_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            mongodb_uri=self.mongodb_uri,
            database_name=self.database_name,
            collection_name=self.collection_name,
            artifact_dir=ARTIFACTS_DIR
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(
            artifact_dir=ARTIFACTS_DIR
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            artifact_dir=ARTIFACTS_DIR
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            artifact_dir=ARTIFACTS_DIR
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        return ModelEvaluationConfig(
            artifact_dir=ARTIFACTS_DIR
        )
