"""Forecasters: naive baselines and a direct multi-horizon LightGBM."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd



RANDOM_STATE = 42


def lgb_params(n_estimators: int = 700) -> dict:
    return {
        "objective": "regression_l1",  # MAE is the operationally relevant loss
        "metric": "mae",
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }


@dataclass
class DirectLightGBM:
    """One model for all horizons, with the horizon passed as a feature.

    Direct multi-horizon avoids the error accumulation of recursive
    one-step-ahead forecasting, which is the main failure mode at 168h.
    """

    n_estimators: int = 700

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "DirectLightGBM":
        self.feature_names_ = [c for c in X.columns if c != "baseline"]
        self.model = lgb.LGBMRegressor(**lgb_params(self.n_estimators))
        fit_kwargs = {}
        if eval_set is not None:
            eval_X, eval_y = eval_set
            fit_kwargs["eval_set"] = [(eval_X[self.feature_names_], eval_y)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self.model.fit(X[self.feature_names_], y, **fit_kwargs)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names_])


def naive_predictions(
    prices: np.ndarray, pos: int, horizons: np.ndarray, season: int
) -> np.ndarray:
    """Seasonal naive: repeat the value from ``season`` hours before the target."""
    idx = pos + horizons - season
    return prices[idx]


def weekly_profile_predictions(
    prices: np.ndarray, pos: int, horizons: np.ndarray, n_weeks: int = 4
) -> np.ndarray:
    """Median of the same local weekday-hour over the previous ``n_weeks`` weeks."""
    idx = np.array([pos + horizons - 168 * w for w in range(1, n_weeks + 1)])
    return np.median(prices[idx], axis=0)
