from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10
    apply_sentiment: bool = True


class SimilarRequest(BaseModel):
    movie_id: int
    top_k: int = 10


class ExplainRequest(BaseModel):
    user_id: int
    top_k: int = 10
