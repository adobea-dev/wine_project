"""
Statistical tests for paired fold metrics.

Uses Wilcoxon signed-rank or paired t-test on per-fold scores.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, ttest_rel


def _paired_vectors(df_long: pd.DataFrame, model_a: str, model_b: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract paired arrays aligned by (repeat, fold).
    """
    a = df_long[df_long["model_name"] == model_a][["repeat", "fold", metric]].copy()
    b = df_long[df_long["model_name"] == model_b][["repeat", "fold", metric]].copy()

    merged = a.merge(b, on=["repeat", "fold"], suffixes=("_a", "_b")).dropna()
    return merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()


def bootstrap_ci_of_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs = x - y
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan

    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(diffs[idx])))

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def compare_two_models(
    df_long: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "ordinal_mae",
    test: str = "wilcoxon",
) -> Dict[str, Any]:
    """
    Returns a compact results dict for reporting.
    Lower metric is better for ordinal_mae and severe_error_rate.
    """
    x, y = _paired_vectors(df_long, model_a, model_b, metric)

    out: Dict[str, Any] = {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_pairs": int(len(x)),
        "mean_a": float(np.mean(x)) if len(x) else np.nan,
        "mean_b": float(np.mean(y)) if len(y) else np.nan,
        "mean_diff_a_minus_b": float(np.mean(x - y)) if len(x) else np.nan,
    }

    ci_lo, ci_hi = bootstrap_ci_of_mean_diff(x, y)
    out["mean_diff_ci_95"] = [ci_lo, ci_hi]

    if len(x) == 0:
        out["test"] = test
        out["statistic"] = np.nan
        out["p_value"] = np.nan
        return out

    if test.lower() == "ttest":
        stat, p = ttest_rel(x, y, nan_policy="omit")
        out["test"] = "paired_t_test"
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
        return out

    # Default Wilcoxon signed-rank
    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    out["test"] = "wilcoxon_signed_rank"
    out["statistic"] = float(stat)
    out["p_value"] = float(p)
    return out
