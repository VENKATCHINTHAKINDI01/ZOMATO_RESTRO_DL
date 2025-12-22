import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


from src.components.model_trainer import DLModelTrainer

trainer = DLModelTrainer()

trainer.initiate_model_training(
    "artifacts/feature_engineering/final_restaurant_features.csv",
    "artifacts/text_vectorization/restaurant_text_vectors.csv",
)
