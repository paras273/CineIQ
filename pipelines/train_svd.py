import logging
from pathlib import Path

import mlflow

from backend.app.config import settings
from backend.app.services.svd_service import train_svd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name="train_svd"):
        metrics = train_svd(Path(settings.processed_data_dir), Path(settings.artifact_dir))
        mlflow.log_params(metrics.get("params", {}))
        mlflow.log_metrics(metrics.get("metrics", {}))
        if metrics.get("artifacts"):
            for path in metrics["artifacts"]:
                mlflow.log_artifact(path)


if __name__ == "__main__":
    main()
