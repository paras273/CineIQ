import ast
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


MOVIELENS_REQUIRED = {"ratings", "movies", "links"}
MOVIELENS_RATINGS_COLS = {"userId", "movieId", "rating"}
MOVIELENS_MOVIES_COLS = {"movieId", "title", "genres"}
MOVIELENS_LINKS_COLS = {"movieId", "tmdbId"}
TMDB_MOVIES_COLS = {"id", "genres"}
TMDB_CREDITS_COLS = {"id", "cast", "crew"}
TMDB_KEYWORDS_COLS = {"id", "keywords"}
IMDB_REVIEW_COLS = {"review", "text"}


def _find_csv_by_keywords(folder: Path, keywords: Iterable[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    for path in folder.glob("*.csv"):
        name = path.name.lower()
        if all(k in name for k in keywords):
            return path
    return None


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def discover_movielens_files(movielens_dir: Path, fallback_dirs: Optional[List[Path]] = None) -> Dict[str, Path]:
    checked_dirs = [movielens_dir]
    if fallback_dirs:
        checked_dirs.extend(fallback_dirs)

    for directory in checked_dirs:
        files = {}
        ratings = _find_csv_by_keywords(directory, ["ratings"])
        movies = _find_csv_by_keywords(directory, ["movies"])
        links = _find_csv_by_keywords(directory, ["links"])
        if ratings:
            files["ratings"] = ratings
        if movies:
            files["movies"] = movies
        if links:
            files["links"] = links
        missing = MOVIELENS_REQUIRED - set(files.keys())
        if not missing:
            return files

    checked_display = ", ".join(str(path) for path in checked_dirs)
    raise FileNotFoundError(
        "Missing MovieLens files. Checked directories: "
        f"{checked_display}. Expected files include ratings.csv, movies.csv, links.csv."
    )


def discover_tmdb_files(tmdb_dir: Path) -> Dict[str, Optional[Path]]:
    files = {
        "movies": _find_csv_by_keywords(tmdb_dir, ["movie"]),
        "credits": _find_csv_by_keywords(tmdb_dir, ["credit"]),
        "keywords": _find_csv_by_keywords(tmdb_dir, ["keyword"]),
    }
    return files


def discover_imdb_reviews_file(imdb_dir: Path) -> Optional[Path]:
    return _find_csv_by_keywords(imdb_dir, ["dataset"]) or _find_csv_by_keywords(
        imdb_dir, ["review"]
    )


def _safe_eval_list(value: str) -> List[dict]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _extract_names(values: List[dict], key: str = "name", limit: int = 6) -> List[str]:
    names = [item.get(key, "").strip() for item in values]
    return [name for name in names if name][:limit]


def parse_genres(genres_value: str) -> List[str]:
    if isinstance(genres_value, str) and "|" in genres_value and "{" not in genres_value:
        return [g.strip() for g in genres_value.split("|") if g.strip()]
    items = _safe_eval_list(genres_value)
    return _extract_names(items)


def parse_keywords(keywords_value: str) -> List[str]:
    items = _safe_eval_list(keywords_value)
    return _extract_names(items)


def parse_cast(cast_value: str, limit: int = 6) -> List[str]:
    items = _safe_eval_list(cast_value)
    return _extract_names(items, limit=limit)


def parse_director(crew_value: str) -> Optional[str]:
    items = _safe_eval_list(crew_value)
    for item in items:
        if item.get("job") == "Director":
            return item.get("name")
    return None


def merge_movielens_tmdb(
    movielens_movies: pd.DataFrame,
    links: pd.DataFrame,
    tmdb_movies: Optional[pd.DataFrame],
    tmdb_credits: Optional[pd.DataFrame],
    tmdb_keywords: Optional[pd.DataFrame],
) -> pd.DataFrame:
    enriched = movielens_movies.merge(links, on="movieId", how="left")
    if tmdb_movies is None:
        return enriched

    tmdb_movies = tmdb_movies.copy()
    tmdb_movies["id"] = pd.to_numeric(tmdb_movies.get("id"), errors="coerce")
    enriched["tmdbId"] = pd.to_numeric(enriched.get("tmdbId"), errors="coerce")

    tmdb_credits = tmdb_credits.copy() if tmdb_credits is not None else None
    tmdb_keywords = tmdb_keywords.copy() if tmdb_keywords is not None else None

    if tmdb_credits is not None:
        tmdb_credits["id"] = pd.to_numeric(tmdb_credits.get("id"), errors="coerce")
    if tmdb_keywords is not None:
        tmdb_keywords["id"] = pd.to_numeric(tmdb_keywords.get("id"), errors="coerce")

    merged = enriched.merge(tmdb_movies, left_on="tmdbId", right_on="id", how="left")
    if tmdb_credits is not None:
        merged = merged.merge(tmdb_credits, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_credits"))
    if tmdb_keywords is not None:
        merged = merged.merge(tmdb_keywords, left_on="tmdbId", right_on="id", how="left", suffixes=("", "_keywords"))

    return merged


def build_movie_metadata(merged: pd.DataFrame) -> pd.DataFrame:
    def _safe_text_col(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna("").astype(str)
        return pd.Series("", index=df.index, dtype="object")

    def _safe_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(pd.NA, index=df.index, dtype="float")

    metadata = pd.DataFrame()
    metadata["movieId"] = merged["movieId"]
    metadata["title"] = merged["title"]

    metadata["genres"] = _safe_text_col(merged, "genres")
    metadata["overview"] = _safe_text_col(merged, "overview")

    metadata["tmdb_genres"] = _safe_text_col(merged, "genres_y")
    if metadata["tmdb_genres"].eq("").all():
        metadata["tmdb_genres"] = _safe_text_col(merged, "genres")
    metadata["tmdb_genres"] = metadata["tmdb_genres"].apply(parse_genres)

    keywords_col = _safe_text_col(merged, "keywords")
    if keywords_col.eq("").all():
        keywords_col = _safe_text_col(merged, "keywords_y")
    metadata["keywords"] = keywords_col.apply(parse_keywords)

    metadata["cast"] = _safe_text_col(merged, "cast").apply(parse_cast)
    metadata["director"] = _safe_text_col(merged, "crew").apply(parse_director)

    release_date = _safe_text_col(merged, "release_date")
    if release_date.eq("").all():
        release_date = _safe_text_col(merged, "releaseDate")
    metadata["release_year"] = pd.to_datetime(release_date, errors="coerce").dt.year
    metadata["decade"] = (metadata["release_year"] // 10) * 10

    metadata["text_features"] = (
        metadata["tmdb_genres"].apply(lambda x: " ".join(x))
        + " "
        + metadata["keywords"].apply(lambda x: " ".join(x))
        + " "
        + metadata["cast"].apply(lambda x: " ".join(x))
        + " "
        + metadata["director"].fillna("")
        + " "
        + metadata["overview"].fillna("")
    )

    return metadata


def load_csv(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)
    return pd.read_csv(path)


def prepare_datasets(
    movielens_dir: Path,
    tmdb_dir: Path,
    imdb_dir: Path,
    fallback_movielens_dirs: Optional[List[Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Path]]:
    ml_files = discover_movielens_files(movielens_dir, fallback_movielens_dirs)
    ratings = load_csv(ml_files["ratings"])
    movies = load_csv(ml_files["movies"])
    links = load_csv(ml_files["links"])

    _validate_columns(ratings, MOVIELENS_RATINGS_COLS, "MovieLens ratings")
    _validate_columns(movies, MOVIELENS_MOVIES_COLS, "MovieLens movies")
    _validate_columns(links, MOVIELENS_LINKS_COLS, "MovieLens links")

    tmdb_files = discover_tmdb_files(tmdb_dir)
    tmdb_movies = load_csv(tmdb_files["movies"]) if tmdb_files.get("movies") else None
    tmdb_credits = load_csv(tmdb_files["credits"]) if tmdb_files.get("credits") else None
    tmdb_keywords = load_csv(tmdb_files["keywords"]) if tmdb_files.get("keywords") else None

    if tmdb_movies is not None:
        _validate_columns(tmdb_movies, TMDB_MOVIES_COLS, "TMDB movies")
    if tmdb_credits is not None:
        _validate_columns(tmdb_credits, TMDB_CREDITS_COLS, "TMDB credits")
    if tmdb_keywords is not None:
        _validate_columns(tmdb_keywords, TMDB_KEYWORDS_COLS, "TMDB keywords")

    merged = merge_movielens_tmdb(movies, links, tmdb_movies, tmdb_credits, tmdb_keywords)
    metadata = build_movie_metadata(merged)

    imdb_path = discover_imdb_reviews_file(imdb_dir)
    if imdb_path is not None:
        imdb_df = load_csv(imdb_path)
        if not (IMDB_REVIEW_COLS & set(imdb_df.columns)):
            raise ValueError("IMDb reviews must include a review or text column")

    return ratings, movies, metadata, imdb_path
