"""
Repeated CV Evaluation

Runs RepeatedStratifiedKFold and computes both standard and ordinal metrics
on the validation fold for each model.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project imports
from ..features import create_interaction_features, select_features_correlation, StandardScalerWrapper


def _ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Ordinal metrics for labels {0,1,2}.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    diffs = np.abs(y_true - y_pred)
    ordinal_mae = float(np.mean(diffs))
    severe_error_rate = float(np.mean(diffs >= 2))   # two-level jump
    within_1_accuracy = float(np.mean(diffs <= 1))   # usually near 1.0 in 3-class

    return {
        "ordinal_mae": ordinal_mae,
        "severe_error_rate": severe_error_rate,
        "within_1_accuracy": within_1_accuracy,
    }


def _standard_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@dataclass
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 42

    # If True, will add interaction features inside each fold (slower).
    create_interactions: bool = False

    # If fixed_features is provided, feature selection is skipped.
    fixed_features: Optional[List[str]] = None

    # Used only if fixed_features is None
    selection_threshold: float = 0.1


def run_repeated_stratified_cv(
    df: pd.DataFrame,
    target_column: str,
    models: Dict[str, Any],
    cfg: Optional[CVConfig] = None,
) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with one row per (model, repeat, fold).

    No leakage:
    - Scaling is fitted on train fold only.
    - If feature selection is used, it is fit on train fold only.

    Fast mode (recommended):
    - Provide cfg.fixed_features (your 41 features) so we do not reselect features each fold.
    """
    if cfg is None:
        cfg = CVConfig()

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y_all = df[target_column].astype(int).values
    df_base = df.copy()

    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state,
    )

    rows: List[Dict[str, Any]] = []
    split_index = 0

    for train_idx, val_idx in rskf.split(df_base, y_all):
        split_index += 1
        repeat_id = (split_index - 1) // cfg.n_splits + 1
        fold_id = (split_index - 1) % cfg.n_splits + 1

        df_train = df_base.iloc[train_idx].copy()
        df_val = df_base.iloc[val_idx].copy()

        if cfg.create_interactions:
            df_train = create_interaction_features(df_train)
            df_val = create_interaction_features(df_val)

        # Choose features
        if cfg.fixed_features is not None and len(cfg.fixed_features) > 0:
            selected_features = [c for c in cfg.fixed_features if c in df_train.columns]
        else:
            selected_features = select_features_correlation(
                df_train,
                target_column=target_column,
                threshold=cfg.selection_threshold,
            )

        if not selected_features:
            raise ValueError("No features selected for CV. Check fixed_features or selection settings.")

        X_train = df_train[selected_features]
        y_train = df_train[target_column].astype(int).values
        X_val = df_val[selected_features]
        y_val = df_val[target_column].astype(int).values

        scaler = StandardScalerWrapper(columns=selected_features)
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        for model_name, base_model in models.items():
            m = clone(base_model)

            try:
                m.fit(X_train_s, y_train)
                y_pred = m.predict(X_val_s)

                std = _standard_metrics(y_val, y_pred)
                ordm = _ordinal_metrics(y_val, y_pred)

                rows.append({
                    "model_name": model_name,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "n_features": int(len(selected_features)),
                    **std,
                    **ordm,
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "model_name": model_name,
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "n_features": int(len(selected_features)),
                    "accuracy": np.nan,
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "f1_macro": np.nan,
                    "ordinal_mae": np.nan,
                    "severe_error_rate": np.nan,
                    "within_1_accuracy": np.nan,
                    "error": str(e),
                })

    return pd.DataFrame(rows)
