import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def _build_mappings(ratings: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    movie_ids = ratings["movieId"].unique().tolist()
    user_ids = ratings["userId"].unique().tolist()
    movie_id_to_idx = {str(mid): idx for idx, mid in enumerate(movie_ids)}
    user_id_to_idx = {str(uid): idx for idx, uid in enumerate(user_ids)}
    return {"movie_id_to_idx": movie_id_to_idx, "user_id_to_idx": user_id_to_idx}


def train_cf_models(processed_dir: Path, artifact_dir: Path) -> Dict:
    ratings_path = processed_dir / "ratings_processed.csv"
    ratings = pd.read_csv(ratings_path)

    mappings = _build_mappings(ratings)
    movie_id_to_idx = mappings["movie_id_to_idx"]
    user_id_to_idx = mappings["user_id_to_idx"]

    num_movies = len(movie_id_to_idx)
    num_users = len(user_id_to_idx)

    row = ratings["movieId"].astype(str).map(movie_id_to_idx)
    col = ratings["userId"].astype(str).map(user_id_to_idx)
    data = ratings["rating"].astype(float)

    item_matrix = csr_matrix((data, (row, col)), shape=(num_movies, num_users))
    user_matrix = item_matrix.T

    item_similarity = cosine_similarity(item_matrix)
    user_similarity = cosine_similarity(user_matrix)

    artifact_dir.joinpath("features").mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("mappings").mkdir(parents=True, exist_ok=True)

    item_path = artifact_dir / "features" / "cf_item_similarity.joblib"
    user_path = artifact_dir / "features" / "cf_user_similarity.joblib"
    joblib.dump(item_similarity, item_path)
    joblib.dump(user_similarity, user_path)

    mapping_path = artifact_dir / "mappings" / "cf_mappings.json"
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mappings, handle)

    return {
        "params": {"similarity": "cosine"},
        "metrics": {},
        "artifacts": [str(item_path), str(user_path), str(mapping_path)],
    }


def _score_items_from_history(item_similarity: np.ndarray, history_idx: List[int], history_ratings: List[float]) -> np.ndarray:
    if not history_idx:
        return np.zeros(item_similarity.shape[0])
    weights = np.array(history_ratings)
    sim_subset = item_similarity[:, history_idx]
    weighted = sim_subset.dot(weights)
    norm = np.abs(weights).sum() if weights.size else 1.0
    return weighted / norm


def recommend_for_user(ratings: pd.DataFrame, item_similarity: np.ndarray, movie_id_to_idx: Dict[str, int], user_id: int, top_k: int) -> List[int]:
    user_ratings = ratings[ratings["userId"] == user_id]
    history_idx = user_ratings["movieId"].astype(str).map(movie_id_to_idx).dropna().astype(int).tolist()
    history_scores = user_ratings["rating"].astype(float).tolist()

    scores = _score_items_from_history(item_similarity, history_idx, history_scores)
    seen = set(user_ratings["movieId"].astype(str))

    ranked_idx = np.argsort(scores)[::-1]
    recommendations = []
    for idx in ranked_idx:
        movie_id = list(movie_id_to_idx.keys())[list(movie_id_to_idx.values()).index(idx)]
        if movie_id not in seen:
            recommendations.append(int(movie_id))
        if len(recommendations) >= top_k:
            break
    return recommendations


def similar_movies(item_similarity: np.ndarray, movie_id_to_idx: Dict[str, int], movie_id: int, top_k: int) -> List[int]:
    idx = movie_id_to_idx.get(str(movie_id))
    if idx is None:
        return []
    scores = item_similarity[idx]
    ranked_idx = np.argsort(scores)[::-1]
    results = []
    for i in ranked_idx:
        if i == idx:
            continue
        movie_id_out = list(movie_id_to_idx.keys())[list(movie_id_to_idx.values()).index(i)]
        results.append(int(movie_id_out))
        if len(results) >= top_k:
            break
    return results


def _score_items_from_users(
    ratings: pd.DataFrame,
    user_similarity: np.ndarray,
    user_id_to_idx: Dict[str, int],
    user_id: int,
    candidate_movie_ids: List[int],
    max_neighbors: int = 50,
) -> Dict[int, float]:
    user_idx = user_id_to_idx.get(str(user_id))
    if user_idx is None:
        return {movie_id: 0.0 for movie_id in candidate_movie_ids}

    sim_scores = user_similarity[user_idx]
    neighbor_indices = np.argsort(sim_scores)[::-1][: max_neighbors + 1]
    neighbor_indices = [idx for idx in neighbor_indices if idx != user_idx]
    neighbor_users = {str(uid): sim_scores[idx] for uid, idx in user_id_to_idx.items() if idx in neighbor_indices}

    output = {}
    for movie_id in candidate_movie_ids:
        movie_ratings = ratings[ratings["movieId"] == movie_id]
        if movie_ratings.empty:
            output[movie_id] = 0.0
            continue
        weighted = 0.0
        norm = 0.0
        for _, row in movie_ratings.iterrows():
            sim = neighbor_users.get(str(row["userId"]))
            if sim is None:
                continue
            weighted += sim * float(row["rating"])
            norm += abs(sim)
        output[movie_id] = weighted / norm if norm > 0 else 0.0
    return output


def score_candidates_for_user(
    ratings: pd.DataFrame,
    item_similarity: np.ndarray,
    user_similarity: np.ndarray,
    movie_id_to_idx: Dict[str, int],
    user_id_to_idx: Dict[str, int],
    user_id: int,
    candidate_movie_ids: List[int],
    item_weight: float = 0.6,
    user_weight: float = 0.4,
) -> Dict[int, float]:
    user_ratings = ratings[ratings["userId"] == user_id]
    history_idx = user_ratings["movieId"].astype(str).map(movie_id_to_idx).dropna().astype(int).tolist()
    history_scores = user_ratings["rating"].astype(float).tolist()
    item_scores = _score_items_from_history(item_similarity, history_idx, history_scores)
    user_scores = _score_items_from_users(
        ratings,
        user_similarity,
        user_id_to_idx,
        user_id,
        candidate_movie_ids,
    )

    output = {}
    for movie_id in candidate_movie_ids:
        idx = movie_id_to_idx.get(str(movie_id))
        item_score = float(item_scores[idx]) if idx is not None else 0.0
        user_score = user_scores.get(movie_id, 0.0)
        output[movie_id] = item_weight * item_score + user_weight * user_score
    return output
