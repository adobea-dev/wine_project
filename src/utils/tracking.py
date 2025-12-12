from __future__ import annotations
import time, json
from pathlib import Path
from typing import Dict, Any
from src.utils.io import write_json, write_yaml   # or use relative import if inside package

def now_ts() -> str:
    """Return a UTC timestamp string for filenames or logs."""
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())

def save_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    """Save evaluation metrics to JSON."""
    write_json(metrics, path)

def save_confusion_matrix(cm: list[list[int]], path: str | Path) -> None:
    """Save confusion matrix to JSON."""
    write_json({"confusion_matrix": cm}, path)

def save_model_metadata(model, name: str, feature_names, label_names, path: str | Path) -> None:
    """Save metadata about the trained model for reproducibility."""
    meta = {
        "model_name": name,
        "class": model.__class__.__name__,
        "params": model.get_params() if hasattr(model, "get_params") else {},
        "feature_names": list(feature_names) if feature_names is not None else None,
        "label_names": list(label_names) if label_names is not None else None,
        "timestamp_utc": now_ts(),
    }
    write_json(meta, path)
