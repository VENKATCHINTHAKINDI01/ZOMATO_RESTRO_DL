import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class DLModelTrainer:
    def __init__(self):
        try:
            self.model_dir = os.path.join(ARTIFACTS_DIR, "model_trainer")
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception as e:
            raise ZomatoException(e, sys)

    def _build_model(self, input_dim: int):
        model = Sequential([
            Dense(128, activation="relu", input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="linear")
        ])

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )
        return model

    def initiate_model_training(self, restaurant_features_path, text_vectors_path):
        try:
            logger.info("Starting DL model training with MLflow")

            # Load data
            tab_df = pd.read_csv(restaurant_features_path)
            text_df = pd.read_csv(text_vectors_path)
            df = pd.merge(tab_df, text_df, on="Name", how="inner")

            y = df["avg_rating"].values
            embed_cols = [c for c in df.columns if c.startswith("w2v_")]
            X = df[embed_cols].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = self._build_model(X_train.shape[1])

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            mlflow.set_experiment("Zomato_Restaurant_Recommender_DL")

            with mlflow.start_run():
                mlflow.log_param("epochs", 30)
                mlflow.log_param("batch_size", 32)
                mlflow.log_param("embedding_dim", X.shape[1])

                model.fit(
                    X_train,
                    y_train,
                    validation_split=0.2,
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1
                )

                # Evaluation
                preds = model.predict(X_test).flatten()

                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                logger.info(
                    f"Evaluation -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, "
                    f"MAE: {mae:.4f}, R2: {r2:.4f}"
                )

                # Save artifacts
                model_path = os.path.join(self.model_dir, "dl_model.keras")
                model.save(model_path)

                scaler_path = os.path.join(self.model_dir, "scaler.pkl")
                joblib.dump(scaler, scaler_path)

                mlflow.log_artifact(model_path)
                mlflow.log_artifact(scaler_path)

                mlflow.tensorflow.log_model(model, artifact_path="dl_model")

            logger.info("DL model training + MLflow logging completed")
            return model_path

        except Exception as e:
            raise ZomatoException(e, sys)
