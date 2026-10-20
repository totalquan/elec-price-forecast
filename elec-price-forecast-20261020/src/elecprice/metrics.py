"""Forecast metrics.

Electricity prices go negative and cross zero, so MAPE is undefined or
meaningless. We report MAE / RMSE plus sMAPE and WAPE, which stay stable
around zero and are the standard choice for power price benchmarking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (0-200). Safe when values are near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.zeros_like(y_true, dtype=float)
    np.divide(np.abs(y_true - y_pred), denom, out=ratio, where=denom >= 1e-9)
    return float(np.mean(ratio) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error (percent), robust to zero crossings."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom < 1e-9:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error (pred - actual). Positive = over-forecast."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE_%": smape(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "n": int(len(y_true)),
    }


def evaluate_by_horizon(pred: pd.DataFrame, group_col: str = "horizon") -> pd.DataFrame:
    """pred must contain columns y_true, y_pred plus group_col."""
    rows = []
    for key, g in pred.groupby(group_col):
        row = evaluate(g["y_true"], g["y_pred"])
        row[group_col] = key
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = [group_col] + [c for c in out.columns if c != group_col]
    return out[cols].sort_values(group_col).reset_index(drop=True)


def evaluate_by_regime(pred: pd.DataFrame) -> pd.DataFrame:
    """Error split by price regime — what traders actually care about."""
    def regime(p: float) -> str:
        if p < 0:
            return "negative"
        if p < 50:
            return "low"
        if p < 150:
            return "normal"
        if p < 300:
            return "high"
        return "spike"

    tmp = pred.copy()
    tmp["regime"] = tmp["y_true"].map(regime)
    rows = []
    for name, g in tmp.groupby("regime"):
        row = evaluate(g["y_true"], g["y_pred"])
        row["regime"] = name
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[["regime"] + [c for c in out.columns if c != "regime"]]
