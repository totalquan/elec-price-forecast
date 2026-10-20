"""Report figures (English labels on purpose — no font dependency)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import PRICE_COL

BLUE = "#1F4E79"
ORANGE = "#C55A11"
GREY = "#7F7F7F"
GREEN = "#2E7D32"


def _style(ax, title: str, ylabel: str = "EUR/MWh"):
    ax.set_title(title, fontsize=11, loc="left", color="#1A1A1A")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def price_overview(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    s = df[PRICE_COL]
    ax.plot(s.index, s.values, color=BLUE, linewidth=0.35, alpha=0.55, label="Hourly price")
    monthly = s.resample("MS").mean()
    ax.plot(monthly.index, monthly.values, color=ORANGE, linewidth=1.6, label="Monthly mean")
    ax.axhline(0, color=GREY, linewidth=0.7, linestyle="--")
    _style(ax, "DE-LU day-ahead price, 2024-01 to 2026-10")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def day_ahead_example(pred: pd.DataFrame, out: Path) -> None:
    lgb = pred[(pred["model"] == "LightGBM") & (pred["horizon"] <= 24)]
    # Pick the most volatile test day — a more informative picture than an average day
    vol = lgb.groupby("origin_ts")["y_true"].std().sort_values()
    origin = vol.index[-3]
    g = lgb[lgb["origin_ts"] == origin].sort_values("horizon")
    naive = pred[
        (pred["model"] == "Naive_168h")
        & (pred["origin_ts"] == origin)
        & (pred["horizon"] <= 24)
    ].sort_values("horizon")

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=150)
    x = g["target_ts"].dt.tz_convert("Europe/Berlin")
    ax.plot(x, g["y_true"], color=BLUE, linewidth=2.0, marker="o", markersize=2.5, label="Actual")
    ax.plot(x, g["y_pred"], color=ORANGE, linewidth=1.8, marker="o", markersize=2.5, label="LightGBM")
    ax.plot(x, naive["y_pred"], color=GREY, linewidth=1.2, linestyle="--", label="Naive 168h")
    _style(ax, f"Day-ahead forecast vs actual (origin {origin:%Y-%m-%d %H:%M} UTC)")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def error_by_hour(pred: pd.DataFrame, out: Path) -> None:
    lgb = pred[(pred["model"] == "LightGBM") & (pred["horizon"] <= 24)].copy()
    lgb["err"] = (lgb["y_pred"] - lgb["y_true"]).abs()
    hourly = lgb.groupby("hour")["err"].mean()

    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=150)
    ax.bar(hourly.index, hourly.values, color=BLUE, width=0.7)
    _style(ax, "Day-ahead MAE by hour of day (local time)")
    ax.set_xlabel("Hour", fontsize=8)
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def week_ahead_example(pred: pd.DataFrame, out: Path) -> None:
    lgb = pred[(pred["model"] == "LightGBM") & (pred["horizon"] <= 168)]
    origins = sorted(lgb["origin_ts"].unique())
    origin = origins[-8]
    g = lgb[lgb["origin_ts"] == origin].sort_values("horizon")
    naive = pred[
        (pred["model"] == "Naive_168h")
        & (pred["origin_ts"] == origin)
        & (pred["horizon"] <= 168)
    ].sort_values("horizon")

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=150)
    x = g["target_ts"].dt.tz_convert("Europe/Berlin")
    ax.plot(x, g["y_true"], color=BLUE, linewidth=1.0, label="Actual")
    ax.plot(x, g["y_pred"], color=ORANGE, linewidth=1.3, label="LightGBM direct 168h")
    ax.plot(x, naive["y_pred"], color=GREY, linewidth=0.9, linestyle="--", label="Naive 168h")
    _style(ax, f"Week-ahead forecast (origin {origin:%Y-%m-%d} UTC)")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def feature_importance(imp: pd.Series, out: Path, top_n: int = 15) -> None:
    top = imp.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.barh(top.index, top.values, color=GREEN)
    _style(ax, "LightGBM feature importance (split gain proxy)", ylabel="")
    ax.set_xlabel("Importance", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def weekly_midterm(wk: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=150)
    ax.plot(wk["week_end"], wk["y_true"], color=BLUE, marker="o", markersize=3, label="Actual")
    ax.plot(wk["week_end"], wk["ridge"], color=ORANGE, marker="s", markersize=3, label="Ridge 1-week ahead")
    ax.plot(wk["week_end"], wk["ma4"], color=GREY, linestyle="--", linewidth=1.1, label="4-week MA")
    _style(ax, "Mid-term: next-week average price, walk-forward")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
