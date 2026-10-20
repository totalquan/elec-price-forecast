"""FastAPI inference service — loads the artifact produced by train_final.py."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .features import build_origin_matrix

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(os.getenv("ELECPRICE_ARTIFACT", ROOT / "artifacts" / "model.joblib"))

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ARTIFACT_PATH.exists():
        STATE["bundle"] = joblib.load(ARTIFACT_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title="Electricity Price Forecast API",
    version="1.0.0",
    description="Day-ahead / week-ahead price forecasting for the DE-LU market.",
    lifespan=lifespan,
)


def _bundle() -> dict:
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"model artifact not found at {ARTIFACT_PATH}; run scripts/train_final.py",
        )
    return bundle


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "bundle" in STATE}


@app.get("/model/info")
def model_info() -> dict:
    bundle = _bundle()
    return {
        "market": bundle["meta"]["market"],
        "unit": bundle["meta"]["unit"],
        "algo": bundle["meta"]["algo"],
        "trained_at": bundle["meta"]["trained_at"],
        "train_range": bundle["meta"]["train_range"],
        "sample_count": bundle["meta"]["sample_count"],
        "features": bundle["feature_cols"],
        "holdout_metrics": bundle["meta"].get("holdout_metrics"),
    }


@app.get("/predict")
def predict(horizon: int = Query(24, ge=1, le=168)) -> dict:
    bundle = _bundle()
    history: pd.DataFrame = bundle["history"]
    if len(history) < 800:
        raise HTTPException(status_code=500, detail="history buffer too short")

    X = build_origin_matrix(history, horizon=horizon)
    # The model predicts the residual against the weekly anchor.
    preds = bundle["model"].predict(X[bundle["feature_cols"]]) + X["baseline"].to_numpy()

    origin_ts = history.index[-1]
    timestamps = pd.date_range(
        origin_ts + pd.Timedelta(hours=1), periods=horizon, freq="h", tz="UTC"
    )
    return {
        "origin_ts": origin_ts.isoformat(),
        "horizon": horizon,
        "unit": bundle["meta"]["unit"],
        "algo": bundle["meta"]["algo"],
        "points": [
            {
                "ts": ts.isoformat(),
                "local_time": ts.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M"),
                "price": round(float(p), 2),
            }
            for ts, p in zip(timestamps, preds)
        ],
        "summary": {
            "mean": round(float(preds.mean()), 2),
            "max": round(float(preds.max()), 2),
            "min": round(float(preds.min()), 2),
            "negative_hours": int((preds < 0).sum()),
        },
    }
