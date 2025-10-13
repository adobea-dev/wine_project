from __future__ import annotations
import json, platform, sys, datetime as dt
import importlib
from pathlib import Path
from typing import Dict, Iterable

DEFAULT_PKGS = [
    "pandas", "numpy", "sklearn", "imblearn",
    "matplotlib", "seaborn", "shap", "xgboost", "lightgbm", "joblib"
]

def collect_runtime_info(extra_packages: Iterable[str] = ()) -> Dict:
    info = {
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {}
    }
    for name in list(DEFAULT_PKGS) + list(extra_packages or []):
        try:
            mod = importlib.import_module(name if name != "sklearn" else "sklearn")
            ver = getattr(mod, "__version__", "unknown")
        except Exception:
            ver = "not_installed"
        info["packages"][name] = ver
    return info

def save_runtime_info(path: str | Path, extra_packages: Iterable[str] = ()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(collect_runtime_info(extra_packages), f, indent=2)