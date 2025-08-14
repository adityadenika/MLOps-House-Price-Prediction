from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

log = get_logger("preprocess")

def build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[Pipeline, List[str], List[str]]:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log.info("Detected columns", extra={"extra":{"numerical": len(num_cols), "categorical": len(cat_cols)}})

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols
