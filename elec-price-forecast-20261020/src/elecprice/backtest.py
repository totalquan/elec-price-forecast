"""Rolling-origin (walk-forward) backtest.

No random splits: a random split on hourly price data leaks future information
through neighbouring hours and through rolling features. Every test origin here
is predicted using only data available at that origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .data import PRICE_COL, to_daily, to_weekly
from .features import (
    available_feature_cols,
    build_direct_dataset,
    build_weekly_dataset,
    calendar_features,
)
from .models import (
    DirectLightGBM,
    naive_predictions,
    weekly_profile_predictions,
)
from .metrics import evaluate, evaluate_by_horizon, evaluate_by_regime

MAX_HORIZON = 168


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics_day_ahead: pd.DataFrame
    metrics_week_ahead: pd.DataFrame
    metrics_by_horizon: pd.DataFrame
    metrics_by_regime: pd.DataFrame
    model: DirectLightGBM
    feature_importance: pd.Series
    train_range: tuple[pd.Timestamp, pd.Timestamp]
    test_range: tuple[pd.Timestamp, pd.Timestamp]
    weekly: dict = field(default_factory=dict)


def _matrix_at(
    feat: pd.DataFrame, prices: np.ndarray, pos: int, horizons: np.ndarray
) -> pd.DataFrame:
    """Feature rows for one origin, repeated over the horizon ladder."""
    row = feat.iloc[[pos]].reset_index(drop=True)
    target_index = feat.index[pos + horizons]
    block = pd.concat([row] * len(horizons), ignore_index=True)
    cal = calendar_features(target_index).reset_index(drop=True)
    for col in cal.columns:
        block[col] = cal[col].to_numpy()
    block["horizon"] = horizons
    block["baseline"] = prices[pos]  # origin price anchor
    return block[available_feature_cols(block) + ["baseline"]]


def run_backtest(
    df: pd.DataFrame,
    test_start: str | pd.Timestamp,
    origin_step: int = 24,
    day_horizon: int = 24,
    week_horizon: int = 168,
) -> BacktestResult:
    from .features import add_price_features

    prices = df[PRICE_COL].to_numpy(dtype=float)
    feat = add_price_features(df)

    test_start = pd.Timestamp(test_start, tz="UTC")
    test_pos = int(df.index.searchsorted(test_start))
    train_end = test_pos - week_horizon
    if train_end < 500:
        raise ValueError("not enough history before test_start")

    # --- training set (origins strictly before the test window) ---
    train_df = df.iloc[: train_end + 1]
    X_tr, y_tr, meta_tr = build_direct_dataset(
        train_df, horizons=range(1, week_horizon + 1), origin_step=origin_step
    )

    # Hold out the last 60 days of the training window for early stopping.
    cut = meta_tr["target_ts"] < (train_df.index[-1] - pd.Timedelta(days=60))
    # Train on the residual against the weekly anchor, not the raw level.
    y_resid = y_tr - X_tr["baseline"]
    model = DirectLightGBM(n_estimators=900).fit(
        X_tr[cut], y_resid[cut], eval_set=(X_tr[~cut], y_resid[~cut])
    )

    # --- rolling test origins: 23:00 local, the last hour known at gate closure ---
    if "local_hour" in df.columns:
        candidates = np.where(df["local_hour"].to_numpy() == 23)[0]
        test_positions = candidates[
            (candidates >= test_pos) & (candidates <= len(df) - week_horizon - 1)
        ]
    else:
        test_positions = np.arange(test_pos, len(df) - week_horizon, origin_step)
    horizons = np.arange(1, week_horizon + 1)

    rows = []
    for pos in test_positions:
        X_test = _matrix_at(feat, prices, pos, horizons)
        y_true = prices[pos + horizons]

        preds = {
            # Change model + origin anchor = final price forecast.
            "LightGBM": model.predict(X_test) + X_test["baseline"].to_numpy(),
            "Persistence": np.full(len(horizons), prices[pos]),
            "Naive_24h": naive_predictions(prices, pos, horizons, 24),
            "Naive_168h": naive_predictions(prices, pos, horizons, 168),
            "WeeklyProfile": weekly_profile_predictions(prices, pos, horizons),
        }
        target_ts = df.index[pos + horizons]
        for name, y_hat in preds.items():
            rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "origin_ts": df.index[pos],
                        "target_ts": target_ts,
                        "horizon": horizons,
                        "y_true": y_true,
                        "y_pred": y_hat,
                    }
                )
            )

    pred = pd.concat(rows, ignore_index=True)
    pred["hour"] = pred["target_ts"].dt.tz_convert("Europe/Berlin").dt.hour

    day = pred[pred["horizon"] <= day_horizon]
    week = pred[pred["horizon"] <= week_horizon]

    metrics_day = _metrics_table(day)
    # Naive_24h needs y[t+h-24]; for h>24 that value is not known at the origin,
    # so it is not a legitimate week-ahead baseline.
    metrics_week = _metrics_table(week, exclude={"Naive_24h"})
    by_h = evaluate_by_horizon(day)
    by_regime = _regime_table(day)

    weekly_mid = run_weekly_midterm(df, n_test_weeks=26)

    return BacktestResult(
        predictions=pred,
        metrics_day_ahead=metrics_day,
        metrics_week_ahead=metrics_week,
        metrics_by_horizon=by_h,
        metrics_by_regime=by_regime,
        model=model,
        feature_importance=model.feature_importances_,
        train_range=(df.index[0], df.index[train_end]),
        test_range=(df.index[test_pos], df.index[-1]),
        weekly=weekly_mid,
    )


def _metrics_table(pred: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for name, g in pred.groupby("model"):
        if exclude and name in exclude:
            continue
        row = evaluate(g["y_true"], g["y_pred"])
        row["model"] = name
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values("MAE").reset_index(drop=True)
    return out[["model", "MAE", "RMSE", "sMAPE_%", "WAPE_%", "bias", "R2", "n"]]


def _regime_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in pred.groupby("model"):
        tab = evaluate_by_regime(g)
        tab.insert(0, "model", name)
        rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def run_weekly_midterm(df: pd.DataFrame, n_test_weeks: int = 26) -> dict:
    """Walk-forward forecast of next-week average price (mid-term, 周度).

    Two references: a 4-week moving average baseline and a Ridge model on
    lagged weekly means plus seasonality. With only ~140 weekly observations,
    a regularised linear model is a more honest choice than a GBDT.
    """
    weekly = to_weekly(df)
    X, y = build_weekly_dataset(weekly)
    first_test = max(len(y) - n_test_weeks, 12)

    records = []
    for i in range(first_test, len(y)):
        if i < 30:
            continue
        anchor = float(y.iloc[i - 1])  # last observed week
        # Model the week-over-week change, not the level: the level drifts.
        delta = y.diff()
        X_train, d_train = X.iloc[:i], delta.iloc[:i]
        ok = d_train.notna()
        model = Ridge(alpha=1.0).fit(X_train[ok], d_train[ok])
        pred = anchor + float(model.predict(X.iloc[[i]])[0])
        baseline = float(np.mean(y.iloc[max(0, i - 4) : i]))
        records.append(
            {
                "week_end": y.index[i],
                "y_true": float(y.iloc[i]),
                "ridge": pred,
                "ma4": baseline,
            }
        )

    wk = pd.DataFrame(records)
    summary = {
        "n_weeks": int(len(wk)),
        "ridge": evaluate(wk["y_true"], wk["ridge"]),
        "ma4": evaluate(wk["y_true"], wk["ma4"]),
        "last_actual": float(wk["y_true"].iloc[-1]),
        "last_pred": float(wk["ridge"].iloc[-1]),
    }
    # Champion / challenger: a simple 4-week mean is hard to beat in a trending
    # market, so it stays champion unless the ML challenger wins out-of-sample.
    champion = "ma4" if summary["ma4"]["MAE"] <= summary["ridge"]["MAE"] else "ridge"
    summary["champion"] = champion
    summary["challenger"] = "ridge" if champion == "ma4" else "ma4"
    summary["champion_last_pred"] = float(wk[champion].iloc[-1])

    monthly = _monthly_midterm(df, n_test_months=12)
    summary["monthly"] = monthly
    return {"weekly_frame": wk, "summary": summary}


def _monthly_midterm(df: pd.DataFrame, n_test_months: int = 12) -> dict:
    """Same logic on monthly averages — small sample, reported with caveats."""
    daily = to_daily(df)
    monthly = daily[PRICE_COL].resample("MS").mean().to_frame().dropna()
    X = pd.DataFrame(index=monthly.index)
    for lag in (1, 2, 3, 12):
        X[f"m_lag_{lag}"] = monthly[PRICE_COL].shift(lag)
    month = monthly.index.month.to_numpy()
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["trend"] = np.arange(len(X), dtype=float)
    y = monthly[PRICE_COL]
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    records = []
    for i in range(max(len(y) - n_test_months, 12), len(y)):
        anchor = float(y.iloc[i - 1])
        delta = y.diff()
        X_train, d_train = X.iloc[:i], delta.iloc[:i]
        ok = d_train.notna()
        model = Ridge(alpha=1.0).fit(X_train[ok], d_train[ok])
        pred = anchor + float(model.predict(X.iloc[[i]])[0])
        records.append(
            {
                "month": y.index[i],
                "y_true": float(y.iloc[i]),
                "pred": pred,
                "last_month": anchor,
                "yoy": float(y.iloc[i - 12]) if i >= 12 else np.nan,
            }
        )

    mf = pd.DataFrame(records)
    metrics = {"ridge": evaluate(mf["y_true"], mf["pred"]) if len(mf) else {}}
    for col in ("last_month", "yoy"):
        sub = mf.dropna(subset=[col])
        if len(sub):
            metrics[col] = evaluate(sub["y_true"], sub[col])
    return {"frame": mf, "metrics": metrics}
