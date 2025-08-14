from src.utils.config import load_config
from src.data.data_loader import load_dataset
from src.models.trainer import train_and_log
from src.utils.logger import get_logger
import os

log = get_logger("train")

def main():
    cfg = load_config()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        log.info("Using external MLflow tracking URI", extra={"extra":{"mlflow_uri": uri}})
    df = load_dataset(cfg["data"]["train_path"])
    model_path, metrics = train_and_log(df, cfg)
    log.info("All done", extra={"extra":{"model_path": model_path, "metrics": metrics}})

if __name__ == "__main__":
    main()
