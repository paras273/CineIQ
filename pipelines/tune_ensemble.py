import logging
from pathlib import Path

import json
import joblib
import mlflow
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.services import cf_service, content_service, svd_service
from backend.app.services.hybrid_service import recommend as hybrid_recommend
from backend.app.services.hybrid_service import tune_ensemble_weights
from backend.app.utils.evaluation import precision_at_k, recall_at_k, map_at_k, ndcg_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_metric_names(metrics: dict, k: int) -> dict:
    if not metrics:
        return {}
    return {key.replace("@k", f"_at_{k}"): value for key, value in metrics.items()}


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name="tune_ensemble"):
        metrics = tune_ensemble_weights(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        eval_metrics = _evaluate_hybrid(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        metrics.setdefault("metrics", {}).update(_sanitize_metric_names(eval_metrics, k=10))
        mlflow.log_params(metrics.get("params", {}))
        mlflow.log_metrics(metrics.get("metrics", {}))
        if metrics.get("artifacts"):
            for path in metrics["artifacts"]:
                mlflow.log_artifact(path)


def _evaluate_hybrid(processed_dir: Path, artifact_dir: Path, sample_users: int = 50, k: int = 10) -> dict:
    ratings = pd.read_csv(processed_dir / "ratings_processed.csv")
    item_similarity = joblib.load(artifact_dir / "features" / "cf_item_similarity.joblib")
    user_similarity = joblib.load(artifact_dir / "features" / "cf_user_similarity.joblib")
    mappings = json.load((artifact_dir / "mappings" / "cf_mappings.json").open("r", encoding="utf-8"))
    movie_id_to_idx = mappings["movie_id_to_idx"]
    user_id_to_idx = mappings["user_id_to_idx"]

    tfidf_matrix = joblib.load(artifact_dir / "features" / "content_tfidf.joblib")
    movie_ids = json.load((artifact_dir / "mappings" / "content_movie_ids.json").open("r", encoding="utf-8"))
    svd_model = joblib.load(artifact_dir / "models" / "svd_model.joblib")

    eligible_users = ratings.groupby("userId").filter(lambda x: len(x) >= 5)["userId"].unique()
    if len(eligible_users) == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "map@k": 0.0, "ndcg@k": 0.0}
    rng = np.random.default_rng(42)
    sampled = rng.choice(eligible_users, size=min(sample_users, len(eligible_users)), replace=False)

    all_movies = ratings["movieId"].unique().tolist()
    precisions, recalls, maps, ndcgs = [], [], [], []
    for user_id in sampled:
        user_ratings = ratings[ratings["userId"] == user_id]
        holdout = int(user_ratings.sample(1, random_state=42)["movieId"].iloc[0])
        history = [mid for mid in user_ratings["movieId"].tolist() if mid != holdout]
        seen = set(user_ratings["movieId"].tolist())
        unseen = [m for m in all_movies if m not in seen]
        if not unseen:
            continue
        sample_size = min(50, len(unseen))
        candidates = [holdout] + rng.choice(unseen, size=sample_size, replace=False).tolist()

        cf_scores = cf_service.score_candidates_for_user(
            ratings,
            item_similarity,
            user_similarity,
            movie_id_to_idx,
            user_id_to_idx,
            int(user_id),
            candidates,
        )
        content_scores = content_service.score_candidates_for_user(tfidf_matrix, movie_ids, candidates, history)
        svd_scores = svd_service.score_candidates_for_user(svd_model, int(user_id), candidates)
        ranked = hybrid_recommend(candidates, cf_scores, content_scores, svd_scores, None, False, k * 5)
        ranked_ids = [item["movie_id"] for item in ranked]

        actual = [holdout]
        precisions.append(precision_at_k(actual, ranked_ids, k))
        recalls.append(recall_at_k(actual, ranked_ids, k))
        maps.append(map_at_k([actual], [ranked_ids], k))
        ndcgs.append(ndcg_at_k(actual, ranked_ids, k))

    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "map@k": float(np.mean(maps)),
        "ndcg@k": float(np.mean(ndcgs)),
    }


if __name__ == "__main__":
    main()
