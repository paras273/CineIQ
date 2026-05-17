from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.config import settings
from backend.app.schemas.responses import UserProfileResponse
from backend.app.services.artifact_loader import ArtifactLoader
from backend.app.services.profile_service import build_user_profile

router = APIRouter()


@lru_cache(maxsize=1)
def _get_loader() -> ArtifactLoader:
    return ArtifactLoader(Path(settings.artifact_dir), Path(settings.processed_data_dir))


@router.get("/user-profile/{user_id}", response_model=UserProfileResponse)
def user_profile(user_id: int) -> UserProfileResponse:
    loader = _get_loader()
    ratings = loader.load_ratings()
    metadata = loader.load_metadata()

    if ratings.empty:
        raise HTTPException(status_code=404, detail="Ratings data not found")

    profile = build_user_profile(ratings, metadata, user_id)
    return UserProfileResponse(**profile)
