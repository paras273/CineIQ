import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


@dataclass(frozen=True)
class Settings:
    data_dir: str = _get_env("DATA_DIR", "data")
    raw_data_dir: str = _get_env("RAW_DATA_DIR", "data/raw")
    movielens_dir: str = _get_env("MOVIELENS_DIR", "data/raw/movielens")
    tmdb_dir: str = _get_env("TMDB_DIR", "data/raw/tmdb")
    imdb_reviews_dir: str = _get_env("IMDB_REVIEWS_DIR", "data/raw/imdb_reviews")
    artifact_dir: str = _get_env("ARTIFACT_DIR", "artifacts")
    processed_data_dir: str = _get_env("PROCESSED_DATA_DIR", "data/processed")
    mlflow_tracking_uri: str = _get_env("MLFLOW_TRACKING_URI", "mlruns")
    api_host: str = _get_env("API_HOST", "0.0.0.0")
    api_port: int = int(_get_env("API_PORT", "8000"))
    sentiment_model: str = _get_env("SENTIMENT_MODEL", "vader")
    hybrid_cf_weight: float = float(_get_env("HYBRID_CF_WEIGHT", "0.4"))
    hybrid_content_weight: float = float(_get_env("HYBRID_CONTENT_WEIGHT", "0.3"))
    hybrid_svd_weight: float = float(_get_env("HYBRID_SVD_WEIGHT", "0.3"))
    top_k_default: int = int(_get_env("TOP_K_DEFAULT", "10"))


settings = Settings()
