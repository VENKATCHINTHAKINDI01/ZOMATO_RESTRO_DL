import os
import sys
import pandas as pd

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class DataTransformation:
    def __init__(self):
        try:
            self.transformation_dir = os.path.join(
                ARTIFACTS_DIR, "data_transformation"
            )
            os.makedirs(self.transformation_dir, exist_ok=True)

            self.reviews_clean_path = os.path.join(
                self.transformation_dir, "reviews_clean.csv"
            )
            self.restaurants_clean_path = os.path.join(
                self.transformation_dir, "restaurants_clean.csv"
            )

        except Exception as e:
            raise ZomatoException(e, sys)

    def initiate_data_transformation(self, ingestion_artifacts: dict):
        """
        Cleans raw datasets and saves transformed versions
        """
        try:
            logger.info("Starting data transformation")

            reviews_path = ingestion_artifacts["reviews_path"]
            restaurants_path = ingestion_artifacts["restaurants_path"]

            reviews = pd.read_csv(reviews_path)
            restaurants = pd.read_csv(restaurants_path)

            # ---------------- Reviews cleaning ----------------
            reviews.dropna(inplace=True)
            reviews = reviews[reviews["Rating"] != "Like"]
            reviews["Rating"] = reviews["Rating"].astype(float)

            reviews.rename(
                columns={"Restaurant": "Name"},
                inplace=True
            )

            # ---------------- Restaurants cleaning ----------------
            restaurants["Cost"] = (
                restaurants["Cost"]
                .astype(str)
                .str.replace(",", "")
                .astype(float)
            )

            # Save transformed data
            reviews.to_csv(self.reviews_clean_path, index=False)
            restaurants.to_csv(self.restaurants_clean_path, index=False)

            logger.info("Data transformation completed successfully")

            return {
                "reviews_clean_path": self.reviews_clean_path,
                "restaurants_clean_path": self.restaurants_clean_path
            }

        except Exception as e:
            raise ZomatoException(e, sys)
