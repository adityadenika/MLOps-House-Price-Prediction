import joblib
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

from src.data.data_preprocessor import build_preprocessor
from src.models.model import build_model
from src.utils.logger import get_logger
from src.utils.config import ensure_dir

log = get_logger("trainer")

def evaluate(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_and_log(df: pd.DataFrame, cfg: dict) -> Tuple[str, Dict[str, float]]:
    target = cfg["project"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        shuffle=cfg["train"].get("shuffle", True),
        random_state=cfg["project"]["seed"]
    )

    preprocessor, *_ = build_preprocessor(df, target)
    model = build_model(cfg["model"]["type"], cfg["model"]["params"])

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    mlflow.set_experiment(cfg["project"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(cfg["model"]["params"])
        log.info("Start training", extra={"extra":{"n_train": len(X_train), "n_valid": len(X_valid)}})

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_valid)
        metrics = evaluate(y_valid, preds)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        ensure_dir(cfg["project"]["artifact_dir"])
        model_path = Path(cfg["project"]["artifact_dir"]) / "model.joblib"
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        log.info("Training finished", extra={"extra":{"metrics": metrics, "artifact": str(model_path)}})
        return str(model_path), metrics
