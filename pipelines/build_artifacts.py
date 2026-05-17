import logging
from pathlib import Path

from backend.app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    artifact_dir = Path(settings.artifact_dir)
    for name in ["processed", "models", "features", "mappings"]:
        path = artifact_dir / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured artifact folder: %s", path)


if __name__ == "__main__":
    main()
