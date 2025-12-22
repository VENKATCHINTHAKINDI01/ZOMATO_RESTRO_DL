import os
import sys
import re
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.logger.logging import get_logger
from src.exception.exception import ZomatoException
from src.constant.application import ARTIFACTS_DIR

logger = get_logger()


class TextVectorization:
    def __init__(self):
        try:
            self.vector_dir = os.path.join(
                ARTIFACTS_DIR, "text_vectorization"
            )
            os.makedirs(self.vector_dir, exist_ok=True)

            self.vector_path = os.path.join(
                self.vector_dir, "restaurant_text_vectors.csv"
            )

            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words("english"))

        except Exception as e:
            raise ZomatoException(e, sys)

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)

        tokens = [
            self.stemmer.stem(word)
            for word in text.split()
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def initiate_text_vectorization(self, feature_artifacts: dict):
        """
        Generates Word2Vec embeddings at restaurant level
        """
        try:
            logger.info("Starting text vectorization (Word2Vec)")

            features_path = feature_artifacts["restaurant_features_path"]
            df = pd.read_csv(features_path)

            # Clean cuisine text
            df["clean_text"] = df["Cuisines"].astype(str).apply(
                self._clean_text
            )

            sentences = df["clean_text"].str.split().tolist()

            # Train Word2Vec
            w2v_model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=1,
                sg=1
            )

            # Convert text to vectors
            def get_vector(tokens):
                vectors = [
                    w2v_model.wv[word]
                    for word in tokens
                    if word in w2v_model.wv
                ]
                if not vectors:
                    return np.zeros(100)
                return np.mean(vectors, axis=0)

            vectors = df["clean_text"].str.split().apply(get_vector)

            vector_df = pd.DataFrame(
                vectors.tolist(),
                columns=[f"w2v_{i}" for i in range(100)]
            )

            final_df = pd.concat(
                [df[["Name"]], vector_df],
                axis=1
            )

            final_df.to_csv(self.vector_path, index=False)

            logger.info("Text vectorization completed successfully")

            return self.vector_path

        except Exception as e:
            raise ZomatoException(e, sys)
