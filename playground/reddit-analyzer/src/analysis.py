from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import hdbscan
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


@dataclass
class ClusterConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_cluster_size: int = 12
    min_samples: int = 4
    max_keywords_per_cluster: int = 200


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb)


def cluster_comments(df: pd.DataFrame, config: ClusterConfig) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["cluster_id"] = -1
        return out

    emb = embed_texts(df["clean_text"].tolist(), config.embedding_model)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(emb)

    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    return out


def extract_cluster_keywords(
    df: pd.DataFrame,
    max_keywords_per_cluster: int = 200,
) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    for cluster_id, part in df.groupby("cluster_id"):
        if int(cluster_id) == -1:
            continue
        texts = part["clean_text"].tolist()
        if len(texts) < 2:
            result[int(cluster_id)] = []
            continue

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            stop_words="english",
        )
        mat = vec.fit_transform(texts)
        mean_scores = np.asarray(mat.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        ranked_idx = np.argsort(-mean_scores)
        top_terms = [
            str(terms[i])
            for i in ranked_idx[:max_keywords_per_cluster]
            if float(mean_scores[i]) > 0
        ]
        result[int(cluster_id)] = top_terms

    return result
