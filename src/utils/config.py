from pathlib import Path
import yaml

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
