import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd


class ArtifactLoader:
    def __init__(self, artifact_dir: Path, processed_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.processed_dir = processed_dir

    def load_ratings(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_dir / "ratings_processed.csv")

    def load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_dir / "movies_metadata.csv")

    def load_movies(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_dir / "movies_processed.csv")

    def load_item_similarity(self):
        return joblib.load(self.artifact_dir / "features" / "cf_item_similarity.joblib")

    def load_user_similarity(self):
        return joblib.load(self.artifact_dir / "features" / "cf_user_similarity.joblib")

    def load_cf_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        mapping_path = self.artifact_dir / "mappings" / "cf_mappings.json"
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["movie_id_to_idx"], payload["user_id_to_idx"]

    def load_content_vectorizer(self):
        return joblib.load(self.artifact_dir / "models" / "content_vectorizer.joblib")

    def load_content_tfidf(self):
        return joblib.load(self.artifact_dir / "features" / "content_tfidf.joblib")

    def load_content_mapping(self) -> List[int]:
        path = self.artifact_dir / "mappings" / "content_movie_ids.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload

    def load_svd_model(self):
        return joblib.load(self.artifact_dir / "models" / "svd_model.joblib")

    def load_svd_mappings(self) -> Dict[str, List[int]]:
        path = self.artifact_dir / "mappings" / "svd_mappings.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_sentiment_scores(self) -> Optional[pd.DataFrame]:
        path = self.artifact_dir / "processed" / "sentiment_scores.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)
