import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from backend.app.config import settings

logger = logging.getLogger(__name__)


def train_or_prepare_sentiment(processed_dir: Path, artifact_dir: Path) -> Dict:
    _ensure_vader_lexicon()
    reviews_path = _find_reviews_file(processed_dir)
    artifact_dir.joinpath("processed").mkdir(parents=True, exist_ok=True)
    sentiment_path = artifact_dir / "processed" / "sentiment_scores.csv"

    if reviews_path is None:
        logger.warning("No review file detected; falling back to metadata overview sentiment")
        grouped, metrics = _sentiment_from_metadata(processed_dir)
        if grouped is None:
            return {"params": {"model": settings.sentiment_model}, "metrics": {}, "artifacts": []}
        grouped.to_csv(sentiment_path, index=False)
        return {
            "params": {"model": settings.sentiment_model, "source": "overview"},
            "metrics": metrics,
            "artifacts": [str(sentiment_path)],
        }

    reviews = pd.read_csv(reviews_path)
    if "review" not in reviews.columns and "text" not in reviews.columns:
        raise ValueError("Reviews file must include review or text column")

    text_col = "review" if "review" in reviews.columns else "text"
    reviews["sentiment"] = _score_texts(reviews[text_col].fillna("").tolist())

    if "movieId" in reviews.columns:
        grouped = reviews.groupby("movieId")["sentiment"].mean().reset_index()
        metrics_source = "reviews_movieId"
    elif "title" in reviews.columns:
        grouped = reviews.groupby("title")["sentiment"].mean().reset_index()
        grouped.rename(columns={"title": "movieTitle"}, inplace=True)
        metrics_source = "reviews_title"
    else:
        logger.warning("Review file lacks movieId/title; falling back to metadata overview sentiment")
        grouped, metrics = _sentiment_from_metadata(processed_dir)
        if grouped is None:
            return {"params": {"model": settings.sentiment_model}, "metrics": {}, "artifacts": []}
        grouped.to_csv(sentiment_path, index=False)
        return {
            "params": {"model": settings.sentiment_model, "source": "overview"},
            "metrics": metrics,
            "artifacts": [str(sentiment_path)],
        }

    grouped.to_csv(sentiment_path, index=False)

    metrics = {
        "num_reviews": int(len(reviews)),
        "avg_sentiment": float(reviews["sentiment"].mean()) if not reviews.empty else 0.0,
        "pos_ratio": float((reviews["sentiment"] > 0).mean()) if not reviews.empty else 0.0,
    }
    if "movieId" in grouped.columns:
        metrics["movies_scored"] = int(grouped["movieId"].nunique())

    return {
        "params": {"model": settings.sentiment_model, "source": metrics_source},
        "metrics": metrics,
        "artifacts": [str(sentiment_path)],
    }


def _find_reviews_file(processed_dir: Path) -> Optional[Path]:
    for name in ["movie_reviews.csv", "imdb_reviews.csv", "reviews.csv"]:
        path = processed_dir / name
        if path.exists():
            return path
    return None


def _ensure_vader_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def get_movie_sentiment(sentiment_df: pd.DataFrame, movie_id: int, title: Optional[str] = None) -> float:
    if sentiment_df is None or sentiment_df.empty:
        return 0.0
    if "movieId" in sentiment_df.columns:
        row = sentiment_df[sentiment_df["movieId"] == movie_id]
        if not row.empty:
            return float(row.iloc[0]["sentiment"])
    if title and "movieTitle" in sentiment_df.columns:
        row = sentiment_df[sentiment_df["movieTitle"] == title]
        if not row.empty:
            return float(row.iloc[0]["sentiment"])
    return 0.0


def rerank_candidates(candidates: List[Dict], sentiment_df: pd.DataFrame) -> List[Dict]:
    if sentiment_df is None or sentiment_df.empty:
        return candidates
    for candidate in candidates:
        sentiment = get_movie_sentiment(sentiment_df, candidate["movie_id"], candidate.get("title"))
        candidate["sentiment"] = sentiment
        boost = sentiment * 0.1
        candidate["score"] += boost
        breakdown = candidate.get("score_breakdown")
        if isinstance(breakdown, dict):
            breakdown["sentiment_boost"] = boost
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def _sentiment_from_metadata(processed_dir: Path) -> Tuple[Optional[pd.DataFrame], Dict]:
    metadata_path = processed_dir / "movies_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "overview" in metadata.columns and metadata["overview"].fillna("").str.strip().any():
            texts = metadata["overview"].fillna("").astype(str).tolist()
            sentiments = _score_texts(texts)
            grouped = pd.DataFrame({"movieId": metadata["movieId"].astype(int), "sentiment": sentiments})
            metrics = {
                "movies_scored": int(len(grouped)),
                "avg_sentiment": float(grouped["sentiment"].mean()) if not grouped.empty else 0.0,
                "pos_ratio": float((grouped["sentiment"] > 0).mean()) if not grouped.empty else 0.0,
            }
            return grouped, metrics

    tmdb_movies_path = _find_tmdb_movies_file(Path(settings.tmdb_dir))
    links_path = Path(settings.movielens_dir) / "links.csv"
    if tmdb_movies_path is None or not links_path.exists():
        logger.warning("TMDB overview fallback unavailable (missing tmdb movies or links)")
        return None, {}

    tmdb_movies = pd.read_csv(tmdb_movies_path)
    if "id" not in tmdb_movies.columns or "overview" not in tmdb_movies.columns:
        logger.warning("TMDB movies file missing id or overview for sentiment fallback")
        return None, {}

    links = pd.read_csv(links_path)
    if "movieId" not in links.columns or "tmdbId" not in links.columns:
        logger.warning("MovieLens links missing movieId/tmdbId for sentiment fallback")
        return None, {}

    tmdb_movies["id"] = pd.to_numeric(tmdb_movies["id"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    merged = links.merge(tmdb_movies[["id", "overview"]], left_on="tmdbId", right_on="id", how="inner")
    if merged.empty:
        logger.warning("No TMDB overview matches found for sentiment fallback")
        return None, {}

    texts = merged["overview"].fillna("").astype(str).tolist()
    sentiments = _score_texts(texts)
    grouped = pd.DataFrame({"movieId": merged["movieId"].astype(int), "sentiment": sentiments})
    metrics = {
        "movies_scored": int(grouped["movieId"].nunique()),
        "avg_sentiment": float(grouped["sentiment"].mean()) if not grouped.empty else 0.0,
        "pos_ratio": float((grouped["sentiment"] > 0).mean()) if not grouped.empty else 0.0,
    }
    return grouped, metrics


def _find_tmdb_movies_file(tmdb_dir: Path) -> Optional[Path]:
    if not tmdb_dir.exists():
        return None
    for path in tmdb_dir.glob("*.csv"):
        name = path.name.lower()
        if "movie" in name:
            return path
    return None


def _score_texts(texts: List[str]) -> List[float]:
    model_name = settings.sentiment_model.lower()
    if model_name == "distilbert":
        scores = _score_with_distilbert(texts)
        if scores is not None:
            return scores
        logger.warning("DistilBERT unavailable; falling back to VADER")
    _ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(str(text))["compound"] for text in texts]


def _score_with_distilbert(texts: List[str]) -> Optional[List[float]]:
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Transformers not available: %s", exc)
        return None
    try:
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        outputs = classifier(texts, truncation=True)
        scores = []
        for output in outputs:
            label = output.get("label", "").upper()
            score = float(output.get("score", 0.0))
            scores.append(score if label == "POSITIVE" else -score)
        return scores
    except Exception as exc:  # pragma: no cover - model download/runtime errors
        logger.warning("DistilBERT inference failed: %s", exc)
        return None
