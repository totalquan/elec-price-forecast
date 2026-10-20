import numpy as np

from elecprice.metrics import evaluate, smape, wape


def test_perfect_prediction():
    y = np.array([10.0, -20.0, 0.0, 150.0])
    m = evaluate(y, y)
    assert m["MAE"] == 0
    assert m["RMSE"] == 0
    assert m["sMAPE_%"] == 0
    assert m["bias"] == 0


def test_smape_handles_zero_and_negative_prices():
    y_true = np.array([0.0, -50.0, 100.0])
    y_pred = np.array([0.0, -40.0, 120.0])
    value = smape(y_true, y_pred)
    assert 0 < value < 200
    assert np.isfinite(value)


def test_wape_is_scale_free():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    assert wape(y_true, y_pred) == 10.0


def test_bias_sign():
    m = evaluate(np.array([100.0]), np.array([120.0]))
    assert m["bias"] == 20.0
