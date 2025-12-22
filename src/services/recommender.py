import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.model_loader import ModelLoader


class RestaurantRecommender:
    def __init__(self):
        loader = ModelLoader()
        self.model = loader.model
        self.scaler = loader.scaler
        self.features_df = loader.features_df
        self.vectors_df = loader.vectors_df

        self.df = pd.merge(
            self.features_df,
            self.vectors_df,
            on="Name",
            how="inner"
        )

        self.embed_cols = [c for c in self.df.columns if c.startswith("w2v_")]

    def recommend(self, cuisine: str, top_n: int = 5):
        # Filter cuisine
        filtered = self.df[
            self.df["Cuisines"].str.contains(cuisine, case=False, na=False)
        ]

        if filtered.empty:
            return []

        X = filtered[self.embed_cols].values
        X_scaled = self.scaler.transform(X)

        # DL predicted scores
        filtered["predicted_score"] = (
            self.model.predict(X_scaled).flatten()
        )

        # Semantic similarity
        sim_matrix = cosine_similarity(X)
        filtered["similarity_score"] = sim_matrix.mean(axis=1)

        # Hybrid score
        filtered["final_score"] = (
            0.7 * filtered["predicted_score"]
            + 0.3 * filtered["similarity_score"]
        )

        result = (
            filtered
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )

        return result[
            ["Name", "Cuisines", "Cost",
             "predicted_score", "similarity_score", "final_score"]
        ].to_dict(orient="records")
