import logging
from pathlib import Path

from backend.app.config import settings
from backend.app.utils.preprocessing import prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    movielens_dir = Path(settings.movielens_dir)
    tmdb_dir = Path(settings.tmdb_dir)
    imdb_dir = Path(settings.imdb_reviews_dir)
    processed_dir = Path(settings.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, metadata, imdb_path = prepare_datasets(
        movielens_dir=movielens_dir,
        tmdb_dir=tmdb_dir,
        imdb_dir=imdb_dir,
        fallback_movielens_dirs=[repo_root],
    )

    ratings_path = processed_dir / "ratings_processed.csv"
    movies_path = processed_dir / "movies_processed.csv"
    metadata_path = processed_dir / "movies_metadata.csv"

    ratings.to_csv(ratings_path, index=False)
    movies.to_csv(movies_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    logger.info("Saved ratings to %s", ratings_path)
    logger.info("Saved movies to %s", movies_path)
    logger.info("Saved metadata to %s", metadata_path)
    if imdb_path:
        logger.info("Detected IMDb reviews at %s", imdb_path)
    else:
        logger.warning("IMDb reviews file not detected; sentiment will use VADER fallback")


if __name__ == "__main__":
    main()
