import logging
from pathlib import Path

import json
import joblib
import mlflow
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.services import content_service
from backend.app.services.content_service import train_content_model
from backend.app.utils.evaluation import precision_at_k, recall_at_k, map_at_k, ndcg_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_metric_names(metrics: dict, k: int) -> dict:
    if not metrics:
        return {}
    return {key.replace("@k", f"_at_{k}"): value for key, value in metrics.items()}


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name="train_content"):
        metrics = train_content_model(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        eval_metrics = _evaluate_content(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        metrics.setdefault("metrics", {}).update(_sanitize_metric_names(eval_metrics, k=10))
        mlflow.log_params(metrics.get("params", {}))
        mlflow.log_metrics(metrics.get("metrics", {}))
        if metrics.get("artifacts"):
            for path in metrics["artifacts"]:
                mlflow.log_artifact(path)


def _evaluate_content(processed_dir: Path, artifact_dir: Path, sample_users: int = 50, k: int = 10) -> dict:
    ratings = pd.read_csv(processed_dir / "ratings_processed.csv")
    tfidf_matrix = joblib.load(artifact_dir / "features" / "content_tfidf.joblib")
    mapping_path = artifact_dir / "mappings" / "content_movie_ids.json"
    movie_ids = json.load(mapping_path.open("r", encoding="utf-8"))

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
        scores = content_service.score_candidates_for_user(tfidf_matrix, movie_ids, candidates, history)
        ranked = [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        actual = [holdout]
        precisions.append(precision_at_k(actual, ranked, k))
        recalls.append(recall_at_k(actual, ranked, k))
        maps.append(map_at_k([actual], [ranked], k))
        ndcgs.append(ndcg_at_k(actual, ranked, k))

    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "map@k": float(np.mean(maps)),
        "ndcg@k": float(np.mean(ndcgs)),
    }


if __name__ == "__main__":
    main()
