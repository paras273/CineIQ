import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def train_content_model(processed_dir: Path, artifact_dir: Path) -> Dict:
    metadata = pd.read_csv(processed_dir / "movies_metadata.csv")
    texts = metadata["text_features"].fillna("")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    artifact_dir.joinpath("models").mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("features").mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("mappings").mkdir(parents=True, exist_ok=True)

    vectorizer_path = artifact_dir / "models" / "content_vectorizer.joblib"
    tfidf_path = artifact_dir / "features" / "content_tfidf.joblib"
    mapping_path = artifact_dir / "mappings" / "content_movie_ids.json"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, tfidf_path)

    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata["movieId"].astype(int).tolist(), handle)

    return {
        "params": {"max_features": 20000},
        "metrics": {},
        "artifacts": [str(vectorizer_path), str(tfidf_path), str(mapping_path)],
    }


def similar_movies(tfidf_matrix, movie_ids: List[int], movie_id: int, top_k: int) -> List[int]:
    if movie_id not in movie_ids:
        return []
    idx = movie_ids.index(movie_id)
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]
    results = []
    for i in ranked:
        if i == idx:
            continue
        results.append(int(movie_ids[i]))
        if len(results) >= top_k:
            break
    return results


def score_candidates_for_user(
    tfidf_matrix,
    movie_ids: List[int],
    candidate_movie_ids: List[int],
    user_history: List[int],
) -> Dict[int, float]:
    if not user_history:
        return {movie_id: 0.0 for movie_id in candidate_movie_ids}
    history_indices = [movie_ids.index(mid) for mid in user_history if mid in movie_ids]
    if not history_indices:
        return {movie_id: 0.0 for movie_id in candidate_movie_ids}
    history_matrix = tfidf_matrix[history_indices]
    scores = linear_kernel(history_matrix, tfidf_matrix).mean(axis=0)

    output = {}
    for movie_id in candidate_movie_ids:
        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            output[movie_id] = float(scores[idx])
        else:
            output[movie_id] = 0.0
    return output


def recommend_from_profile(
    tfidf_matrix,
    movie_ids: List[int],
    profile_vector: np.ndarray,
    top_k: int,
) -> List[int]:
    scores = linear_kernel(profile_vector, tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]
    results = []
    for idx in ranked:
        results.append(int(movie_ids[idx]))
        if len(results) >= top_k:
            break
    return results
