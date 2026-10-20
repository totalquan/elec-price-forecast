import numpy as np
import pandas as pd
import pytest

from elecprice.features import (
    FEATURE_COLS,
    PRICE_FEATURES,
    add_price_features,
    build_direct_dataset,
)


def synthetic_prices(n: int = 1500) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    price = 80 + 30 * np.sin(2 * np.pi * t / 24) + 15 * np.sin(2 * np.pi * t / 168)
    local = idx.tz_convert("Europe/Berlin")
    return pd.DataFrame(
        {
            "price": price,
            "local_hour": local.hour,
            "local_dow": local.dayofweek,
            "local_month": local.month,
            "local_date": local.normalize(),
            "is_weekend": (local.dayofweek >= 5).astype(int),
        },
        index=idx,
    )


def test_features_do_not_leak_future_prices():
    """Changing prices AFTER the origin must not change features AT the origin."""
    df = synthetic_prices()
    pos = 900
    before = add_price_features(df).iloc[pos][PRICE_FEATURES].to_numpy(dtype=float)

    tampered = df.copy()
    tampered.iloc[pos + 1 :, tampered.columns.get_loc("price")] += 500.0
    after = add_price_features(tampered).iloc[pos][PRICE_FEATURES].to_numpy(dtype=float)

    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_direct_dataset_alignment():
    df = synthetic_prices(1200)
    X, y, meta = build_direct_dataset(df, horizons=[1, 24], origin_step=50, origin_start=400)

    assert len(X) == len(y) == len(meta)
    assert set(meta["horizon"].unique()) == {1, 24}

    expected = meta["origin_ts"] + pd.to_timedelta(meta["horizon"], unit="h")
    assert (expected == meta["target_ts"]).all()
    np.testing.assert_allclose(y.to_numpy(), df["price"].loc[meta["target_ts"]].to_numpy())

    assert list(X.columns) == FEATURE_COLS + ["baseline"]
    assert not X.isna().any().any()


def test_calendar_features_match_target_time():
    df = synthetic_prices(1200)
    X, _, meta = build_direct_dataset(df, horizons=[24], origin_step=100, origin_start=400)
    expected_hour = meta["target_ts"].dt.tz_convert("Europe/Berlin").dt.hour.to_numpy()
    np.testing.assert_array_equal(X["target_hour"].to_numpy(), expected_hour)


@pytest.mark.parametrize("horizon", [1, 24, 168])
def test_origin_matrix_shape(horizon):
    from elecprice.features import build_origin_matrix

    df = synthetic_prices(1200)
    X = build_origin_matrix(df, horizon=horizon)
    assert len(X) == horizon
    assert list(X.columns) == FEATURE_COLS + ["baseline"]
    assert X["horizon"].tolist() == list(range(1, horizon + 1))
