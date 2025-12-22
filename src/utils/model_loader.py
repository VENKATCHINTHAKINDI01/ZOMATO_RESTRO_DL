import os
import joblib
import pandas as pd
import tensorflow as tf

from src.constant.application import ARTIFACTS_DIR


class ModelLoader:
    def __init__(self):
        self.model_path = os.path.join(
            ARTIFACTS_DIR, "model_trainer", "dl_model.keras"
        )
        self.scaler_path = os.path.join(
            ARTIFACTS_DIR, "model_trainer", "scaler.pkl"
        )
        self.features_path = os.path.join(
            ARTIFACTS_DIR, "feature_engineering", "restaurant_features.csv"
        )
        self.vectors_path = os.path.join(
            ARTIFACTS_DIR, "text_vectorization", "restaurant_text_vectors.csv"
        )
        self.model = tf.keras.models.load_model(self.model_path,compile=False)
        self.scaler = joblib.load(self.scaler_path)

        self.features_df = pd.read_csv(self.features_path)
        self.vectors_df = pd.read_csv(self.vectors_path)
