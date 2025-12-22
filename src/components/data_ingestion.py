import os
import sys
import pandas as pd

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_dir = os.path.join(
                ARTIFACTS_DIR, "data_ingestion"
            )
            os.makedirs(self.ingestion_dir, exist_ok=True)

            self.reviews_file_path = os.path.join(
                self.ingestion_dir, "reviews.csv"
            )
            self.restaurants_file_path = os.path.join(
                self.ingestion_dir, "restaurants.csv"
            )

        except Exception as e:
            raise ZomatoException(e, sys)

    def initiate_data_ingestion(self):
        """
        Reads raw datasets and stores them into artifacts folder
        """
        try:
            logger.info("Starting data ingestion")

            # ---- SOURCE FILE PATHS (RAW DATA) ----
            raw_reviews_path = os.path.join(
                "zomato_data", "Restaurant reviews.csv"
            )
            raw_restaurants_path = os.path.join(
                "zomato_data", "Restaurant names and Metadata.csv"
            )

            # ---- READ DATA ----
            reviews_df = pd.read_csv(raw_reviews_path)
            restaurants_df = pd.read_csv(raw_restaurants_path)

            logger.info(
                f"Reviews dataset shape: {reviews_df.shape}"
            )
            logger.info(
                f"Restaurants dataset shape: {restaurants_df.shape}"
            )

            # ---- SAVE TO ARTIFACTS ----
            reviews_df.to_csv(
                self.reviews_file_path, index=False
            )
            restaurants_df.to_csv(
                self.restaurants_file_path, index=False
            )

            logger.info("Data ingestion completed successfully")

            return {
                "reviews_path": self.reviews_file_path,
                "restaurants_path": self.restaurants_file_path
            }

        except Exception as e:
            raise ZomatoException(e, sys)
