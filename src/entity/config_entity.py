from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    mongodb_uri: str
    database_name: str
    reviews_collection: str
    metadata_collection: str
    artifact_dir: str


@dataclass
class DataValidationConfig:
    artifact_dir: str


@dataclass
class DataTransformationConfig:
    artifact_dir: str


@dataclass
class ModelTrainerConfig:
    artifact_dir: str


@dataclass
class ModelEvaluationConfig:
    artifact_dir: str
