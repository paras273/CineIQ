import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from sklearn.metrics.pairwise import linear_kernel

from backend.app.config import settings
from backend.app.schemas.requests import ExplainRequest, RecommendRequest, SimilarRequest
from backend.app.schemas.responses import ExplainResponse, RecommendResponse, RecommendationItem, SimilarItem, SimilarResponse
from backend.app.services import cf_service, content_service, svd_service
from backend.app.services.artifact_loader import ArtifactLoader
from backend.app.services.explanations import build_explanations
from backend.app.services.hybrid_service import recommend as hybrid_recommend

router = APIRouter()


@lru_cache(maxsize=1)
def _get_loader() -> ArtifactLoader:
    return ArtifactLoader(Path(settings.artifact_dir), Path(settings.processed_data_dir))


def _parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [v.strip() for v in value.strip("[]").replace("'", "").split(",") if v.strip()]
    return []


def _build_candidate_pool(ratings, metadata, tfidf_matrix, movie_ids, user_id: int, top_k: int) -> List[int]:
    user_history = ratings[ratings["userId"] == user_id]["movieId"].astype(int).tolist()
    if not user_history:
        return metadata["movieId"].astype(int).head(top_k * 5).tolist()

    content_scores = content_service.score_candidates_for_user(tfidf_matrix, movie_ids, movie_ids, user_history)
    ranked = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in ranked[: top_k * 5]]


def _map_title(metadata, movie_id: int) -> str:
    row = metadata[metadata["movieId"] == movie_id]
    if row.empty:
        return "Unknown"
    return str(row.iloc[0]["title"])


def _build_recommendations(user_id: int, top_k: int, apply_sentiment: bool) -> List[RecommendationItem]:
    loader = _get_loader()
    ratings = loader.load_ratings()
    metadata = loader.load_metadata()
    sentiment_df = loader.load_sentiment_scores()

    if ratings.empty or metadata.empty:
        raise HTTPException(status_code=404, detail="Processed data not found")

    item_similarity = loader.load_item_similarity()
    user_similarity = loader.load_user_similarity()
    movie_id_to_idx, user_id_to_idx = loader.load_cf_mappings()

    tfidf_matrix = loader.load_content_tfidf()
    movie_ids = loader.load_content_mapping()

    svd_model = loader.load_svd_model()
    svd_mappings = loader.load_svd_mappings()

    candidate_ids = _build_candidate_pool(ratings, metadata, tfidf_matrix, movie_ids, user_id, top_k)

    cf_scores = cf_service.score_candidates_for_user(
        ratings,
        item_similarity,
        user_similarity,
        movie_id_to_idx,
        user_id_to_idx,
        user_id,
        candidate_ids,
    )
    content_scores = content_service.score_candidates_for_user(tfidf_matrix, movie_ids, candidate_ids, ratings[ratings["userId"] == user_id]["movieId"].astype(int).tolist())
    svd_known = set(svd_mappings.get("movie_ids", []))
    svd_candidates = [mid for mid in candidate_ids if mid in svd_known]
    svd_scores = svd_service.score_candidates_for_user(svd_model, user_id, svd_candidates)
    for mid in candidate_ids:
        svd_scores.setdefault(mid, 0.0)

    ranked = hybrid_recommend(
        candidate_ids,
        cf_scores,
        content_scores,
        svd_scores,
        sentiment_df,
        apply_sentiment,
        top_k,
    )

    items = []
    for rec in ranked:
        movie_id = rec["movie_id"]
        row_df = metadata[metadata["movieId"] == movie_id]
        if row_df.empty:
            continue
        row = row_df.iloc[0].to_dict()
        row["tmdb_genres"] = _parse_list(row.get("tmdb_genres"))
        row["cast"] = _parse_list(row.get("cast"))
        reasons = build_explanations(row, rec["score_breakdown"])
        items.append(
            RecommendationItem(
                movie_id=movie_id,
                title=_map_title(metadata, movie_id),
                score=rec["score"],
                reasons=reasons,
                score_breakdown=rec["score_breakdown"],
            )
        )
    return items


@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    items = _build_recommendations(request.user_id, request.top_k, request.apply_sentiment)
    return RecommendResponse(user_id=request.user_id, recommendations=items)


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    items = _build_recommendations(request.user_id, request.top_k, apply_sentiment=True)
    return ExplainResponse(user_id=request.user_id, explanations=items)


@router.post("/similar", response_model=SimilarResponse)
def similar(request: SimilarRequest) -> SimilarResponse:
    loader = _get_loader()
    metadata = loader.load_metadata()
    if metadata.empty:
        raise HTTPException(status_code=404, detail="Metadata not found")

    item_similarity = loader.load_item_similarity()
    movie_id_to_idx, _ = loader.load_cf_mappings()
    tfidf_matrix = loader.load_content_tfidf()
    movie_ids = loader.load_content_mapping()

    cf_similar = cf_service.similar_movies(item_similarity, movie_id_to_idx, request.movie_id, request.top_k)
    content_similar = content_service.similar_movies(tfidf_matrix, movie_ids, request.movie_id, request.top_k)

    combined = list(dict.fromkeys(cf_similar + content_similar))[: request.top_k]
    scores = []
    cf_scores = {}
    content_scores = {}
    idx = movie_id_to_idx.get(str(request.movie_id))
    if idx is not None:
        sim_row = item_similarity[idx]
        for mid in combined:
            mid_idx = movie_id_to_idx.get(str(mid))
            if mid_idx is not None:
                cf_scores[mid] = float(sim_row[mid_idx])
    if request.movie_id in movie_ids:
        content_idx = movie_ids.index(request.movie_id)
        content_row = linear_kernel(tfidf_matrix[content_idx], tfidf_matrix).flatten()
        for mid in combined:
            if mid in movie_ids:
                content_scores[mid] = float(content_row[movie_ids.index(mid)])

    for mid in combined:
        score = 0.5 * cf_scores.get(mid, 0.0) + 0.5 * content_scores.get(mid, 0.0)
        scores.append(SimilarItem(movie_id=mid, title=_map_title(metadata, mid), score=score))

    items = scores

    return SimilarResponse(movie_id=request.movie_id, similar_movies=items)
