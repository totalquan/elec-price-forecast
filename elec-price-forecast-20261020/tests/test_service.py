from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="run scripts/train_final.py first"
)


def test_predict_endpoint():
    from fastapi.testclient import TestClient

    from elecprice.service import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"

        resp = client.get("/predict", params={"horizon": 24})
        assert resp.status_code == 200
        body = resp.json()
        assert body["horizon"] == 24
        assert len(body["points"]) == 24
        assert body["unit"] == "EUR/MWh"
        assert all("ts" in p and "price" in p for p in body["points"])


def test_horizon_validation():
    from fastapi.testclient import TestClient

    from elecprice.service import app

    with TestClient(app) as client:
        assert client.get("/predict", params={"horizon": 0}).status_code == 422
        assert client.get("/predict", params={"horizon": 999}).status_code == 422
