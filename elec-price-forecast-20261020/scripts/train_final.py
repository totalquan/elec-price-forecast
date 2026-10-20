"""Train the final model on all available history and persist the serving bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from elecprice.data import (  # noqa: E402
    attach_fundamentals,
    load_prices,
    parse_public_power_json,
)
from elecprice.features import build_direct_dataset  # noqa: E402
from elecprice.metrics import evaluate  # noqa: E402
from elecprice.models import DirectLightGBM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/de_lu_price_raw.json"))
    ap.add_argument(
        "--fundamentals", default=str(ROOT / "data/raw/de_public_power_raw.json")
    )
    ap.add_argument("--artifact", default=str(ROOT / "artifacts/model.joblib"))
    ap.add_argument("--horizon", type=int, default=168)
    args = ap.parse_args()

    df = load_prices(args.data)
    fund_path = Path(args.fundamentals)
    if fund_path.exists():
        df = attach_fundamentals(df, parse_public_power_json(fund_path))
    X, y, meta = build_direct_dataset(
        df, horizons=range(1, args.horizon + 1), origin_step=24
    )

    # Honest holdout: last 30 days of targets are never used for fitting.
    cutoff = df.index[-1] - pd.Timedelta(days=30)
    train_mask = meta["target_ts"] < cutoff
    y_resid = y - X["baseline"]
    model = DirectLightGBM(n_estimators=900).fit(X[train_mask], y_resid[train_mask])

    holdout_mask = ~train_mask
    holdout_pred = model.predict(X[holdout_mask]) + X.loc[holdout_mask, "baseline"].to_numpy()
    holdout = evaluate(y[holdout_mask], holdout_pred)
    da_mask = holdout_mask & (meta["horizon"] <= 24)
    holdout_da = evaluate(
        y[da_mask], model.predict(X[da_mask]) + X.loc[da_mask, "baseline"].to_numpy()
    )

    # Refit on everything for serving.
    final = DirectLightGBM(n_estimators=900).fit(X, y_resid)

    bundle = {
        "model": final.model,
        "feature_cols": final.feature_names_,
        "history": df.tail(1200).copy(),
        "meta": {
            "market": "DE-LU day-ahead",
            "unit": "EUR/MWh",
            "algo": "LightGBM direct multi-horizon (h=1..%d)" % args.horizon,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_range": [str(df.index[0]), str(df.index[-1])],
            "sample_count": int(len(X)),
            "holdout_metrics": {
                "all_horizons": to_round(holdout),
                "day_ahead": to_round(holdout_da),
                "holdout_days": 30,
            },
        },
    }

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact)

    (artifact.parent / "meta.json").write_text(
        json.dumps(bundle["meta"], indent=2), encoding="utf-8"
    )
    print(f"trained on {len(X):,} samples -> {artifact}")
    print(json.dumps(bundle["meta"]["holdout_metrics"], indent=2))


def to_round(d: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


if __name__ == "__main__":
    main()
