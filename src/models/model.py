from sklearn.ensemble import RandomForestRegressor

def build_model(kind: str, params: dict):
    kind = (kind or "random_forest").lower()
    if kind == "random_forest":
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported model type: {kind}")
