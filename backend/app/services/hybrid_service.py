from typing import Dict, List

import numpy as np

from backend.app.config import settings
from backend.app.services.sentiment_service import rerank_candidates


def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    min_v, max_v = float(values.min()), float(values.max())
    if max_v - min_v == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def recommend(
    candidate_ids: List[int],
    cf_scores: Dict[int, float],
    content_scores: Dict[int, float],
    svd_scores: Dict[int, float],
    sentiment_df,
    apply_sentiment: bool,
    top_k: int,
) -> List[Dict]:
    cf_norm = _normalize(cf_scores)
    content_norm = _normalize(content_scores)
    svd_norm = _normalize(svd_scores)

    scored = []
    for movie_id in candidate_ids:
        score = (
            settings.hybrid_cf_weight * cf_norm.get(movie_id, 0.0)
            + settings.hybrid_content_weight * content_norm.get(movie_id, 0.0)
            + settings.hybrid_svd_weight * svd_norm.get(movie_id, 0.0)
        )
        scored.append(
            {
                "movie_id": movie_id,
                "score": score,
                "score_breakdown": {
                    "cf": cf_norm.get(movie_id, 0.0),
                    "content": content_norm.get(movie_id, 0.0),
                    "svd": svd_norm.get(movie_id, 0.0),
                    "sentiment_boost": 0.0,
                },
            }
        )

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[: top_k * 2]
    if apply_sentiment:
        scored = rerank_candidates(scored, sentiment_df)
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def tune_ensemble_weights(processed_dir, artifact_dir) -> Dict:
    return {
        "params": {
            "cf_weight": settings.hybrid_cf_weight,
            "content_weight": settings.hybrid_content_weight,
            "svd_weight": settings.hybrid_svd_weight,
        },
        "metrics": {},
        "artifacts": [],
    }
