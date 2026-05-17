from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


def train_svd(processed_dir: Path, artifact_dir: Path) -> Dict:
    ratings = pd.read_csv(processed_dir / "ratings_processed.csv")
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    artifact_dir.joinpath("models").mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("mappings").mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "models" / "svd_model.joblib"
    joblib.dump(model, model_path)

    mappings = {
        "user_ids": ratings["userId"].astype(int).unique().tolist(),
        "movie_ids": ratings["movieId"].astype(int).unique().tolist(),
    }
    mapping_path = artifact_dir / "mappings" / "svd_mappings.json"
    mapping_path.write_text(json_dump(mappings), encoding="utf-8")

    return {
        "params": {"n_factors": 100},
        "metrics": {"rmse": rmse, "mae": mae},
        "artifacts": [str(model_path), str(mapping_path)],
    }


def json_dump(payload: Dict) -> str:
    import json

    return json.dumps(payload)


def predict_user_item_score(model: SVD, user_id: int, movie_id: int) -> float:
    prediction = model.predict(user_id, movie_id)
    return float(prediction.est)


def score_candidates_for_user(model: SVD, user_id: int, candidate_movie_ids: List[int]) -> Dict[int, float]:
    scores = {}
    trainset = getattr(model, "trainset", None)
    for movie_id in candidate_movie_ids:
        if trainset is not None:
            try:
                _ = trainset.to_inner_uid(user_id)
                _ = trainset.to_inner_iid(movie_id)
            except ValueError:
                scores[movie_id] = 0.0
                continue
        scores[movie_id] = predict_user_item_score(model, user_id, movie_id)
    return scores
