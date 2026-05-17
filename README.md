# CINEIQ Movie Recommender

An open, explainable movie recommendation engine that blends collaborative filtering (item-based + user-based), content-based similarity, and Surprise SVD with an optional sentiment-aware reranker. FastAPI serves the model, and Streamlit renders a taste dashboard with Plotly.

This codebase uses the AnantSabharwal/recommender_system logic as the functional base and refactors toward a gurezende-style backend/frontend split.

## Architecture

- backend/app: FastAPI app, services, and schemas
- frontend/streamlit_app: Streamlit dashboard calling FastAPI
- pipelines: data preparation and training scripts with MLflow logging
- artifacts: persisted models, features, and mappings
- data: raw, interim, processed datasets

## Dataset Placement

Place datasets under (preferred):

- data/raw/movielens/ (ratings.csv, movies.csv, links.csv)
- data/raw/tmdb/ (TMDB metadata CSVs: movies, credits, keywords, overview)
- data/raw/imdb_reviews/ (IMDB Dataset.csv or similar)

Legacy layout also supported:

- ratings.csv, movies.csv, links.csv in the repo root

The pipeline auto-detects likely filenames and fails with clear errors if missing.

Expected schemas:

- MovieLens ratings: userId, movieId, rating
- MovieLens movies: movieId, title, genres
- MovieLens links: movieId, tmdbId
- TMDB movies: id, genres (overview recommended)
- TMDB credits: id, cast, crew
- TMDB keywords: id, keywords
- IMDb reviews: review or text column (movieId or title recommended)

## Setup

Create a virtual environment and install dependencies (NumPy is pinned to <2 for scikit-surprise compatibility):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy .env.example to .env and adjust paths if needed.

## Quick Start (Release Artifacts)

This repo is designed to run without local training by using pretrained artifacts.

1. Download `artifacts.zip` from GitHub Releases.
2. Extract it into the repo root so the folder layout matches:

```
artifacts/
  models/
  features/
  mappings/
  processed/
data/
  processed/
```

The app performs a startup validation and will stop with a clear error if any required
artifacts or processed files are missing.

Note: This repo includes `data/processed` files for inference. Raw datasets are excluded.

## Train From Scratch (Optional)

## Preprocessing

```bash
python -m pipelines.prepare_data
```

Outputs are written to data/processed and artifacts/processed.

## Training

```bash
python -m pipelines.train_cf
python -m pipelines.train_content
python -m pipelines.train_svd
python -m pipelines.train_sentiment
python -m pipelines.tune_ensemble
```

MLflow logs to the local mlruns directory by default.
Ranking metrics logged: precision@k, recall@k, map@k, ndcg@k. SVD logs RMSE and MAE.

## Run MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

## Run FastAPI

```bash
uvicorn backend.app.main:app --reload
```

## Run Streamlit

```bash
streamlit run frontend/streamlit_app/app.py
```

## Docker

````bash
docker compose -f docker/docker-compose.yml up --build

## GitHub Push Steps

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
````

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click "New app" and select the repo and branch.
3. Set the app file to:

```
frontend/streamlit_app/app.py
```

4. Add secrets/env vars for the API endpoint (required if FastAPI runs elsewhere):

```
API_URL=https://<your-fastapi-host>
```

5. Deploy.

Note: Streamlit Cloud requires the repo to be hosted on GitHub. If the app fails to start,
download `artifacts.zip` from Releases and extract it into the repo root before deploying.

````

## API Endpoints

- GET /health
- POST /recommend {user_id, top_k, apply_sentiment}
- POST /similar {movie_id, top_k}
- POST /explain {user_id, top_k}
- GET /user-profile/{user_id}

Example request:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"top_k":10,"apply_sentiment":true}'
````

## Known Limitations

- Movie-level sentiment requires a review dataset with movie identifiers or titles.
- Content and CF similarity matrices can be large; adjust max_features if needed.
  If review data is missing, sentiment scores are skipped and recommendations fall back to the base hybrid model.

Sentiment modes:

- Default: VADER
- Optional: DistilBERT (SST-2) when SENTIMENT_MODEL=distilbert; falls back to VADER if model download fails

## Future Work

- Add LIME explanations for content and hybrid contributions.
- Incremental retraining and feedback loops.
