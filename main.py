import sys

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.components.text_vectorizer import TextVectorization
from src.components.model_trainer import DLModelTrainer



logger = get_logger()


def main():
    try:
        logger.info("========== ZOMATO RECOMMENDER PIPELINE STARTED ==========")

        # 1. Data Ingestion
        logger.info("Starting Data Ingestion")
        ingestion = DataIngestion()
        ingestion_artifacts = ingestion.initiate_data_ingestion()

        # 2. Data Validation
        logger.info("Starting Data Validation")
        validation = DataValidation()
        validation.validate_reviews(
            ingestion_artifacts["reviews_path"]
        )
        validation.validate_restaurants(
            ingestion_artifacts["restaurants_path"]
        )

        # 3. Data Transformation
        logger.info("Starting Data Transformation")
        transformer = DataTransformation()
        transformed_artifacts = transformer.initiate_data_transformation(
            ingestion_artifacts
        )

        # 4. Feature Engineering
        logger.info("Starting Feature Engineering")
        fe = FeatureEngineering()
        feature_artifacts = fe.initiate_feature_engineering(
            transformed_artifacts
        )

        # 5. Text Vectorization (Word2Vec)
        logger.info("Starting Text Vectorization")
        vectorizer = TextVectorization()
        text_vector_path = vectorizer.initiate_text_vectorization(
            feature_artifacts
        )

        # 6. Model Training + MLflow
        logger.info("Starting Model Training with MLflow")
        trainer = DLModelTrainer()
        trainer.initiate_model_training(
            restaurant_features_path=feature_artifacts["restaurant_features_path"],
            text_vectors_path=text_vector_path
        )

        logger.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")

    except Exception as e:
        logger.error("Pipeline failed")
        raise ZomatoException(e, sys)


if __name__ == "__main__":
    main()
