from pathlib import Path

from fastapi import FastAPI

from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_profile import router as profile_router
from backend.app.api.routes_recommend import router as recommend_router
from backend.app.config import settings
from backend.app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="CINEIQ Movie Recommender")
app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(profile_router)


def _validate_required_files() -> None:
	base_dir = Path(__file__).resolve().parents[2]
	artifact_dir = base_dir / settings.artifact_dir
	processed_dir = base_dir / settings.processed_data_dir

	required_paths = [
		artifact_dir / "models" / "svd_model.joblib",
		artifact_dir / "models" / "content_vectorizer.joblib",
		artifact_dir / "features" / "cf_item_similarity.joblib",
		artifact_dir / "features" / "cf_user_similarity.joblib",
		artifact_dir / "features" / "content_tfidf.joblib",
		artifact_dir / "mappings" / "cf_mappings.json",
		artifact_dir / "mappings" / "content_movie_ids.json",
		artifact_dir / "mappings" / "svd_mappings.json",
		processed_dir / "movies_metadata.csv",
		processed_dir / "movies_processed.csv",
		processed_dir / "ratings_processed.csv",
	]

	missing = [str(path) for path in required_paths if not path.exists()]
	if missing:
		missing_list = "\n".join(f"- {path}" for path in missing)
		raise RuntimeError(
			"Missing required artifacts or processed data files. "
			"Download artifacts.zip from GitHub Releases and extract it into the repo root, "
			"and ensure data/processed files are present.\n" + missing_list
		)


@app.on_event("startup")
def _startup_checks() -> None:
	_validate_required_files()
