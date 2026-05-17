import logging
from pathlib import Path

import mlflow

from backend.app.config import settings
from backend.app.services.sentiment_service import train_or_prepare_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name="train_sentiment"):
        metrics = train_or_prepare_sentiment(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        mlflow.log_params(metrics.get("params", {}))
        mlflow.log_metrics(metrics.get("metrics", {}))
        if metrics.get("artifacts"):
            for path in metrics["artifacts"]:
                mlflow.log_artifact(path)


if __name__ == "__main__":
    main()
