# src/evaluation/ordinal_metrics.py
import numpy as np

def ordinal_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(np.abs(y_true - y_pred)))

def severe_error_rate(y_true, y_pred) -> float:
    # For your 3-class setup, severe means Low<->High (distance 2)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(np.abs(y_true - y_pred) >= 2))

def within_one_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(np.abs(y_true - y_pred) <= 1))
