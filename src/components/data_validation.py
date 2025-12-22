import os
import sys
import json
import pandas as pd

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class DataValidation:
    def __init__(self):
        try:
            self.validation_dir = os.path.join(
                ARTIFACTS_DIR, "data_validation"
            )
            os.makedirs(self.validation_dir, exist_ok=True)
        except Exception as e:
            raise ZomatoException(e, sys)

    def _validate_dataset(
        self,
        df: pd.DataFrame,
        required_columns: list,
        report_name: str,
    ) -> str:
        try:
            logger.info(f"Starting validation for {report_name}")

            report = {}

            # Basic checks
            report["row_count"] = int(df.shape[0])
            report["column_count"] = int(df.shape[1])
            report["missing_values"] = (
                df.isnull().sum().to_dict()
            )

            # Required column check
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]
            report["missing_required_columns"] = missing_columns

            # Save report
            report_path = os.path.join(
                self.validation_dir, report_name
            )

            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(
                f"Validation report saved at {report_path}"
            )

            return report_path

        except Exception as e:
            raise ZomatoException(e, sys)

    def validate_reviews(self, reviews_path: str) -> str:
        try:
            reviews_df = pd.read_csv(reviews_path)

            required_columns = [
                "Restaurant",
                "Rating",
                "Metadata",
                "Time",
            ]

            return self._validate_dataset(
                reviews_df,
                required_columns,
                "reviews_validation_report.json",
            )

        except Exception as e:
            raise ZomatoException(e, sys)

    def validate_restaurants(self, restaurants_path: str) -> str:
        try:
            restaurants_df = pd.read_csv(restaurants_path)

            required_columns = [
                "Name",
                "Cuisine",
                "Cost",
            ]

            return self._validate_dataset(
                restaurants_df,
                required_columns,
                "restaurants_validation_report.json",
            )

        except Exception as e:
            raise ZomatoException(e, sys)
