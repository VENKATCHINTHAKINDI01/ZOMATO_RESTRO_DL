import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from src.exception.exception import modeltrainexception
from src.logger.logging import logging
from src.logger.logging import get_logger

# Load environment variables
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

ca = certifi.where()


class ZomatoDataExtract:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca
            )
            logging.info("MongoDB connection established successfully")
        except Exception as e:
            raise modeltrainexception(e, sys)

    def csv_to_json_convertor(self, file_path: str):
        try:
            logging.info(f"Reading CSV file: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = json.loads(data.to_json(orient="records"))
            return records
        except Exception as e:
            raise modeltrainexception(e, sys)

    def insert_data_mongodb(self, records, collection_name: str):
        try:
            db = self.client[DATABASE_NAME]
            collection = db[collection_name]

            collection.insert_many(records)

            logging.info(
                f"Inserted {len(records)} records into collection {collection_name}"
            )
            return len(records)
        except Exception as e:
            raise modeltrainexception(e, sys)


if __name__ == "__main__":

    # Correct file paths (SEPARATE DATASETS)
    REVIEWS_FILE_PATH = os.path.join(
        "zomato_data", "Restaurant reviews.csv"
    )

    METADATA_FILE_PATH = os.path.join(
        "zomato_data", "Restaurant names and Metadata.csv"
    )

    REVIEWS_COLLECTION = "restaurant_reviews"
    METADATA_COLLECTION = "restaurant_metadata"

    zomato = ZomatoDataExtract()

    # ---- Reviews dataset ----
    reviews_records = zomato.csv_to_json_convertor(REVIEWS_FILE_PATH)
    zomato.insert_data_mongodb(reviews_records, REVIEWS_COLLECTION)

    # ---- Metadata dataset ----
    metadata_records = zomato.csv_to_json_convertor(METADATA_FILE_PATH)
    zomato.insert_data_mongodb(metadata_records, METADATA_COLLECTION)
    
    logger = get_logger()
    logger.info("Both datasets inserted successfully into MongoDB")


    
