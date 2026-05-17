from functools import lru_cache
from backend.app.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
