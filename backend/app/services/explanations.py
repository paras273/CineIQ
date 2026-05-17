from typing import Dict, List


def build_explanations(metadata_row: Dict, score_breakdown: Dict) -> List[str]:
    reasons = []
    genres = metadata_row.get("tmdb_genres") or []
    director = metadata_row.get("director")
    cast = metadata_row.get("cast") or []

    if genres:
        reasons.append(f"Matches your preference for {genres[0].lower()} movies.")
    if director:
        reasons.append(f"Shares director patterns with titles you enjoy ({director}).")
    if cast:
        reasons.append(f"Features cast you have engaged with ({cast[0]}).")
    if score_breakdown.get("svd", 0) > 0:
        reasons.append("Strong latent match from collaborative signals.")
    sentiment_boost = score_breakdown.get("sentiment_boost", 0)
    if sentiment_boost > 0:
        reasons.append("Audience sentiment is strongly positive.")
    elif sentiment_boost < 0:
        reasons.append("Audience sentiment is mixed to negative.")

    if not reasons:
        reasons.append("Recommended based on your viewing history and ensemble signals.")

    return reasons[:3]
