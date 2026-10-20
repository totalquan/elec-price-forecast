"""Data loading and cleaning for day-ahead electricity prices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOCAL_TZ = "Europe/Berlin"
PRICE_COL = "price"


def parse_energy_charts_json(path: str | Path) -> pd.DataFrame:
    """Parse the open energy-charts / SMARD price response into a UTC-indexed frame.

    The raw feed mixes hourly and 15-minute resolutions, so we normalise to an
    hourly grid right away (mean aggregation).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ts = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({PRICE_COL: pd.to_numeric(pd.Series(raw["price"]), errors="coerce")})
    df.index = pd.DatetimeIndex(ts, name="ts_utc")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    hourly = df[PRICE_COL].resample("1h").mean().to_frame()
    # Very short gaps only: interpolate up to 3 hours, then edge fill.
    hourly[PRICE_COL] = hourly[PRICE_COL].interpolate(limit=3).ffill().bfill()
    return add_local_time(hourly)


def add_local_time(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Attach local-time calendar columns.

    The index stays in UTC (unique, DST-safe); calendar features come from the
    local market timezone, which is what drives the daily price shape.
    """
    out = df.copy()
    local = out.index.tz_convert(tz)
    out["local_hour"] = local.hour
    out["local_dow"] = local.dayofweek
    out["local_month"] = local.month
    out["local_date"] = local.normalize()
    out["is_weekend"] = (out["local_dow"] >= 5).astype(int)
    return out


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean price (used for mid/long-term aggregation)."""
    daily = df[PRICE_COL].groupby(df["local_date"]).mean().to_frame()
    daily.index.name = "date"
    return daily


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly mean price, weeks anchored on Monday."""
    daily = to_daily(df)
    weekly = daily[PRICE_COL].resample("W-SUN").mean().to_frame()
    weekly.index.name = "week_end"
    return weekly


def load_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return parse_energy_charts_json(path)
    df = pd.read_csv(path, parse_dates=["ts_utc"], index_col="ts_utc")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return add_local_time(df.sort_index())


FUNDAMENTAL_MAP = {
    "Load": "load_mw",
    "Residual load": "residual_mw",
    "Solar": "solar_mw",
    "Wind onshore": "wind_onshore_mw",
    "Wind offshore": "wind_offshore_mw",
    "Renewable share of load": "renewable_share_pct",
}


def parse_public_power_json(path: str | Path) -> pd.DataFrame:
    """Parse the open generation/load feed into an hourly UTC frame.

    Residual load (load minus wind and solar) is the main fundamental driver of
    day-ahead prices, so it is worth carrying alongside the price series.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ts = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)

    series: dict[str, pd.Series] = {}
    for item in raw.get("production_types", []):
        name = item.get("name")
        if name in FUNDAMENTAL_MAP:
            values = pd.to_numeric(pd.Series(item["data"]), errors="coerce")
            series[FUNDAMENTAL_MAP[name]] = pd.Series(values.to_numpy(), index=ts)

    if not series:
        raise ValueError("no recognised generation types in feed")

    df = pd.DataFrame(series).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    hourly = df.resample("1h").mean().interpolate(limit=3).ffill().bfill()
    hourly.index.name = "ts_utc"
    return hourly


def attach_fundamentals(
    prices: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    """Left-join fundamentals onto the price frame on the UTC hour."""
    out = prices.join(fundamentals, how="left")
    cols = [c for c in fundamentals.columns if c in out.columns]
    out[cols] = out[cols].ffill(limit=6).bfill(limit=6)
    return out
