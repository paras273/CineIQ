from typing import Dict, List, Optional

from pydantic import BaseModel


class ScoreBreakdown(BaseModel):
    cf: float
    content: float
    svd: float
    sentiment_boost: float = 0.0


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    score: float
    reasons: List[str]
    score_breakdown: ScoreBreakdown


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]


class SimilarItem(BaseModel):
    movie_id: int
    title: str
    score: float


class SimilarResponse(BaseModel):
    movie_id: int
    similar_movies: List[SimilarItem]


class ExplainResponse(BaseModel):
    user_id: int
    explanations: List[RecommendationItem]


class UserProfileResponse(BaseModel):
    user_id: int
    genres: Dict[str, int]
    decades: Dict[str, int]
    directors: Dict[str, int]
    actors: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
