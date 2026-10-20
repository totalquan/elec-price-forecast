"""Run the rolling-origin backtest and write reports + figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from elecprice import plots  # noqa: E402
from elecprice.backtest import run_backtest  # noqa: E402
from elecprice.data import attach_fundamentals, load_prices, parse_public_power_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/de_lu_price_raw.json"))
    ap.add_argument(
        "--fundamentals", default=str(ROOT / "data/raw/de_public_power_raw.json")
    )
    ap.add_argument("--test-start", default="2026-07-20")
    ap.add_argument("--outdir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_prices(args.data)
    fund_path = Path(args.fundamentals)
    if fund_path.exists():
        df = attach_fundamentals(df, parse_public_power_json(fund_path))
        print("fundamentals attached: residual load, load, renewables share")
    else:
        print("no fundamentals file — running price-only")

    print(f"data: {len(df):,} hourly rows, {df.index[0]} -> {df.index[-1]}")
    print(f"test window starts: {args.test_start}\n")

    res = run_backtest(df, test_start=args.test_start)

    fmt = lambda t: t.to_string(index=False, float_format=lambda v: f"{v:,.2f}")
    print("=== Day-ahead (h=1..24) ===")
    print(fmt(res.metrics_day_ahead))
    print("\n=== Week-ahead (h=1..168) ===")
    print(fmt(res.metrics_week_ahead))
    print("\n=== Error by price regime (day-ahead, LightGBM) ===")
    sub = res.metrics_by_regime[res.metrics_by_regime["model"] == "LightGBM"]
    print(fmt(sub.drop(columns=["model"])))
    print("\n=== Mid-term weekly ===")
    print(json.dumps(res.weekly["summary"], indent=2, default=str))

    res.metrics_day_ahead.to_csv(outdir / "metrics_day_ahead.csv", index=False)
    res.metrics_week_ahead.to_csv(outdir / "metrics_week_ahead.csv", index=False)
    res.metrics_by_horizon.to_csv(outdir / "metrics_by_horizon.csv", index=False)
    res.metrics_by_regime.to_csv(outdir / "metrics_by_regime.csv", index=False)
    res.predictions.to_csv(outdir / "predictions.csv", index=False)
    res.weekly["weekly_frame"].to_csv(outdir / "weekly_midterm.csv", index=False)

    plots.price_overview(df, outdir / "price_overview.png")
    plots.day_ahead_example(res.predictions, outdir / "day_ahead_example.png")
    plots.error_by_hour(res.predictions, outdir / "error_by_hour.png")
    plots.week_ahead_example(res.predictions, outdir / "week_ahead_example.png")
    plots.feature_importance(res.feature_importance, outdir / "feature_importance.png")
    plots.weekly_midterm(res.weekly["weekly_frame"], outdir / "weekly_midterm.png")

    summary = {
        "train_range": [str(res.train_range[0]), str(res.train_range[1])],
        "test_range": [str(res.test_range[0]), str(res.test_range[1])],
        "day_ahead": res.metrics_day_ahead.to_dict(orient="records"),
        "week_ahead": res.metrics_week_ahead.to_dict(orient="records"),
        "midterm_weekly": res.weekly["summary"],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nreports written to {outdir}")


if __name__ == "__main__":
    main()
