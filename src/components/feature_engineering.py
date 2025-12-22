import os
import sys
import pandas as pd

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class FeatureEngineering:
    def __init__(self):
        try:
            self.feature_dir = os.path.join(
                ARTIFACTS_DIR, "feature_engineering"
            )
            os.makedirs(self.feature_dir, exist_ok=True)

            self.restaurant_features_path = os.path.join(
                self.feature_dir, "restaurant_features.csv"
            )

        except Exception as e:
            raise ZomatoException(e, sys)

    def initiate_feature_engineering(self, transformed_artifacts: dict):
        """
        Creates restaurant-level features for modeling
        """
        try:
            logger.info("Starting feature engineering")

            reviews_path = transformed_artifacts["reviews_clean_path"]
            restaurants_path = transformed_artifacts["restaurants_clean_path"]

            reviews = pd.read_csv(reviews_path)
            restaurants = pd.read_csv(restaurants_path)

            # ---------------- Merge ----------------
            df = pd.merge(
                reviews,
                restaurants,
                how="left",
                on="Name"
            )

            # ---------------- Aggregate to restaurant level ----------------
            restaurant_df = df.groupby("Name").agg({
                "Rating": "mean",
                "Cost": "first",
                "Cuisines": "first"
            }).reset_index()

            restaurant_df.rename(
                columns={"Rating": "avg_rating"},
                inplace=True
            )

            restaurant_df.to_csv(
                self.restaurant_features_path,
                index=False
            )

            logger.info("Feature engineering completed successfully")

            return {
                "restaurant_features_path": self.restaurant_features_path
            }

        except Exception as e:
            raise ZomatoException(e, sys)
