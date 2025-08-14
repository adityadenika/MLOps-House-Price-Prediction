import pandas as pd
from src.utils.logger import get_logger

log = get_logger("data_loader")

def load_dataset(train_path: str):
    log.info("Loading training dataset", extra={"extra":{"path": train_path}})
    df = pd.read_csv(train_path)
    return df
