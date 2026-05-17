from typing import Dict, List

import pandas as pd
import re


def build_user_profile(ratings: pd.DataFrame, metadata: pd.DataFrame, user_id: int) -> Dict:
    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        return {"user_id": user_id, "genres": {}, "decades": {}, "directors": {}, "actors": {}}

    merged = user_ratings.merge(metadata, on="movieId", how="left")

    if "decade" not in merged.columns or merged["decade"].dropna().empty:
        merged = merged.copy()
        merged["release_year"] = merged.get("release_year")
        merged["release_year"] = merged["release_year"].fillna(
            merged["title"].apply(_extract_year_from_title)
        )
        merged["decade"] = (merged["release_year"] // 10) * 10

    genre_counts = _count_list_column(merged, "tmdb_genres")
    decade_counts = merged["decade"].dropna().value_counts().to_dict()
    director_counts = merged["director"].dropna().value_counts().to_dict()
    actor_counts = _count_list_column(merged, "cast")

    return {
        "user_id": user_id,
        "genres": genre_counts,
        "decades": {str(k): int(v) for k, v in decade_counts.items()},
        "directors": director_counts,
        "actors": actor_counts,
    }


def _count_list_column(df: pd.DataFrame, column: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for values in df[column].dropna():
        if isinstance(values, str):
            values = [v.strip() for v in values.strip("[]").replace("'", "").split(",") if v.strip()]
        for item in values:
            counts[item] = counts.get(item, 0) + 1
    return counts


def _extract_year_from_title(title: str) -> float:
    if not isinstance(title, str):
        return float("nan")
    match = re.search(r"\((\d{4})\)", title)
    if not match:
        return float("nan")
    return float(match.group(1))
