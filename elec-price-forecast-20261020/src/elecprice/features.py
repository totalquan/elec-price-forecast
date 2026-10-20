"""Feature engineering with an explicit day-ahead information set.

Design rule: at forecasting origin ``t`` we only use prices up to ``t``.
Every price-derived feature is therefore shifted by at least one full day,
so the model never sees a value that would not be published before the
day-ahead auction gate closure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COL = "price"

# Lags (hours) of the price level at the origin.
PRICE_LAGS: tuple[int, ...] = (24, 25, 48, 72, 168, 336)

# Calendar features of the TARGET timestamp.
CALENDAR_FEATURES = [
    "target_hour",
    "target_dow",
    "target_month",
    "target_is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

PRICE_FEATURES: list[str] = [
    "gap_lag_24",
    "gap_lag_48",
    "gap_lag_72",
    "gap_lag_168",
    "gap_lag_336",
    "gap_roll_24",
    "gap_roll_168",
    "std_24",
    "std_168",
    "gap_same_hour_last_week",
    "momentum",
]

FEATURE_COLS: list[str] = PRICE_FEATURES + CALENDAR_FEATURES + ["horizon"]

# Optional fundamentals (residual load etc.) — used when present in the frame.
EXOG_FEATURES: list[str] = [
    "residual_mw_lag24",
    "residual_vs_7d",
    "load_vs_7d",
    "ren_share_lag24",
]


def available_feature_cols(frame) -> list[str]:
    """Base features plus whatever fundamentals the frame actually carries."""
    return [c for c in FEATURE_COLS + EXOG_FEATURES if c in frame.columns]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add *relative* price features that respect the day-ahead cutoff.

    Everything is expressed as a gap to the price observed at the origin
    (``price[t]``), not as an absolute level. This matters: day-ahead prices
    drift by a factor of two across seasons, and tree models cannot
    extrapolate an absolute level they have never seen. Gaps stay stationary.
    """
    out = df.copy()
    price = out[PRICE_COL]
    anchor = price  # known at the origin

    for lag in PRICE_LAGS:
        out[f"gap_lag_{lag}"] = price.shift(lag) - anchor

    # Rolling statistics on data available 24h before the origin.
    known = price.shift(24)
    roll24 = known.rolling(24).mean()
    roll168 = known.rolling(168).mean()

    out["gap_roll_24"] = roll24 - anchor
    out["gap_roll_168"] = roll168 - anchor
    out["std_24"] = known.rolling(24).std()
    out["std_168"] = known.rolling(168).std()
    out["gap_same_hour_last_week"] = price.shift(168) - anchor
    out["momentum"] = roll24 - roll168

    # Fundamentals, lagged 24h: at the day-ahead gate we do not know today's
    # residual load, only yesterday's. Deviations from the 7-day norm are more
    # informative than the raw MW level.
    if "residual_mw" in out.columns:
        resid = out["residual_mw"]
        load = out["load_mw"]
        resid_known = resid.shift(24)
        load_known = load.shift(24)
        out["residual_mw_lag24"] = resid_known
        out["residual_vs_7d"] = resid_known - resid_known.rolling(168).mean()
        out["load_vs_7d"] = load_known - load_known.rolling(168).mean()
        if "renewable_share_pct" in out.columns:
            out["ren_share_lag24"] = out["renewable_share_pct"].shift(24)
    return out


def calendar_features(index: pd.DatetimeIndex, tz: str = "Europe/Berlin") -> pd.DataFrame:
    local = index.tz_convert(tz)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    return pd.DataFrame(
        {
            "target_hour": hour,
            "target_dow": dow,
            "target_month": month,
            "target_is_weekend": (dow >= 5).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def build_direct_dataset(
    df: pd.DataFrame,
    horizons: range | list[int] = range(1, 25),
    origin_step: int = 24,
    origin_start: int = 400,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build a direct multi-horizon dataset.

    Returns
    -------
    X : feature matrix (one row per origin/horizon pair)
    y : target price at ``origin + horizon``
    meta : origin timestamp, target timestamp, horizon

    Direct (rather than recursive) multi-step forecasting avoids compounding
    one-step errors, which matters a lot for the 168h horizon.

    The frame also carries a ``baseline`` column: the price observed at the
    origin (known at gate closure). Models are trained on the *change* from
    this anchor, so the target is stationary even when the price level trends.
    """
    feat = add_price_features(df)
    prices = df[PRICE_COL].to_numpy(dtype=float)
    last_origin = len(df) - max(horizons) - 1

    # Align origins to 23:00 local: the last price known at the D-1 noon gate
    # closure, so h=1 maps to 00:00 of the delivery day. Selecting by local
    # hour (not by fixed UTC steps) keeps this correct across DST changes.
    if "local_hour" in df.columns:
        candidates = np.where(df["local_hour"].to_numpy() == 23)[0]
        candidates = candidates[(candidates >= origin_start) & (candidates <= last_origin)]
        positions = candidates[:: max(1, origin_step // 24)]
    else:
        positions = np.arange(origin_start, last_origin + 1, origin_step)

    x_blocks: list[pd.DataFrame] = []
    y_blocks: list[pd.Series] = []
    meta_blocks: list[pd.DataFrame] = []

    for h in horizons:
        cols = [c for c in PRICE_FEATURES + EXOG_FEATURES if c in feat.columns]
        block = feat.iloc[positions][cols].copy()
        target_index = df.index[positions + h]

        cal = calendar_features(target_index)
        block = block.reset_index(drop=True)
        cal = cal.reset_index(drop=True)
        block[CALENDAR_FEATURES] = cal[CALENDAR_FEATURES].to_numpy()
        block["horizon"] = h
        block["baseline"] = prices[positions]  # price at the origin = anchor

        x_blocks.append(block)
        y_blocks.append(pd.Series(prices[positions + h]))
        meta_blocks.append(
            pd.DataFrame(
                {
                    "origin_ts": df.index[positions],
                    "target_ts": target_index,
                    "horizon": h,
                    "baseline": prices[positions],
                }
            )
        )

    X = pd.concat(x_blocks, ignore_index=True)
    X = X[available_feature_cols(X) + ["baseline"]]
    y = pd.concat(y_blocks, ignore_index=True)
    meta = pd.concat(meta_blocks, ignore_index=True)
    return X, y, meta


def build_origin_matrix(
    history: pd.DataFrame,
    horizon: int = 24,
    tz: str = "Europe/Berlin",
    last_origin_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Feature matrix for a single live origin — used by the inference service.

    ``history`` must end at (or after) the origin timestamp. The origin defaults
    to the last observation in ``history``.
    """
    feat = add_price_features(history)
    if last_origin_ts is None:
        row = feat.iloc[[-1]]
        origin_ts = feat.index[-1]
    else:
        row = feat.loc[[last_origin_ts]]
        origin_ts = last_origin_ts

    future_index = pd.date_range(
        origin_ts + pd.Timedelta(hours=1), periods=horizon, freq="h", tz="UTC"
    )
    base = row.reset_index(drop=True)
    block = pd.concat([base] * horizon, ignore_index=True)
    cal = calendar_features(future_index, tz=tz).reset_index(drop=True)
    block[CALENDAR_FEATURES] = cal[CALENDAR_FEATURES].to_numpy()
    block["horizon"] = np.arange(1, horizon + 1)
    # Anchor: the last observed price, i.e. the origin price (known).
    block["baseline"] = float(history[PRICE_COL].iloc[-1])
    return block[available_feature_cols(block) + ["baseline"]]


def build_weekly_dataset(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Supervised set for mid-term (next-week average price) forecasting."""
    s = weekly[PRICE_COL].copy()
    X = pd.DataFrame(index=s.index)
    for lag in (1, 2, 3, 4, 8):
        X[f"w_lag_{lag}"] = s.shift(lag)
    X["w_roll4_mean"] = s.shift(1).rolling(4).mean()
    X["w_roll4_std"] = s.shift(1).rolling(4).std()
    X["w_roll12_mean"] = s.shift(1).rolling(12).mean()
    month = s.index.month.to_numpy()
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["trend"] = np.arange(len(s), dtype=float)
    y = s.copy()
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]
