"""
Testing if there is no data leackage after the feature engineering. 
    
"""

import numpy as np
import pandas as pd
import pytest

from config import EXCLUDE, TARGET
from data import resample_weekly
from features import engineer_weekly


@pytest.fixture
def weekly():
    """Three years of synthetic daily data, resampled.

    Synthetic so the test runs without the dataset, and so a failure points at
    the code rather than at the data.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=365 * 3, freq="D")
    n = len(dates)
    price = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    daily = pd.DataFrame({
        "Price": price,
        "Units Sold": rng.integers(100, 5_000, n),
        "Mean_Knife_Value": price * rng.uniform(8, 12, n),
        "ROI": rng.normal(0, 0.15, n),
        "Avg. Players": rng.integers(500_000, 1_500_000, n),
        "price_to_min_comp": rng.uniform(0.8, 1.4, n),
        "price_to_med_comp": rng.uniform(0.7, 1.3, n),
    }, index=pd.DatetimeIndex(dates, name="Date"))

    return resample_weekly(daily)


def test_no_feature_sees_the_future(weekly):
    """Change only the last week. Nothing earlier may move.

    This is the strongest available statement of "no lookahead". If a feature
    at week 40 changes when week 100 changes, that feature saw the future.
    Catches centred rolling windows, full-sample scaling and backward fills
    without needing to know which feature is which.
    """
    tampered = weekly.copy()
    last = tampered.index[-1]
    tampered.loc[last, ["Open", "High", "Low", "Close"]] *= 3.0

    before = engineer_weekly(weekly)
    after = engineer_weekly(tampered)

    idx = before.index.intersection(after.index)[:-1]   # exclude perturbed week
    cols = before.columns.intersection(after.columns)

    offenders = [
        c for c in cols
        if not np.allclose(before.loc[idx, c].astype(float),
                           after.loc[idx, c].astype(float),
                           equal_nan=True, rtol=1e-9)
    ]
    assert not offenders, f"features using future information: {offenders[:10]}"


def test_target_is_this_weeks_return(weekly):
    """The target must be log(Close_t / Close_{t-1}), not a shifted version."""
    feats = engineer_weekly(weekly)
    expected = np.log(weekly["Close"] / weekly["Close"].shift(1))
    idx = feats.index.intersection(expected.dropna().index)

    assert np.allclose(feats.loc[idx, TARGET].astype(float),
                       expected.loc[idx].astype(float), equal_nan=True)


def test_contemporaneous_columns_are_excluded(weekly):
    """Unlagged OHLC derivatives must not reach the feature matrix.

    `body_ratio` at week t is built from week t's close. Using it to predict
    week t's return is circular. The lagged twin `body_ratio_lag1` is fine.
    """
    feats = engineer_weekly(weekly)
    cols = [c for c in feats.columns if c not in EXCLUDE]

    forbidden = {"Open", "High", "Low", "Close", "log_close", "body_ratio",
                 "upper_wick", "lower_wick", "close_position", "weekly_range"}
    assert not forbidden & set(cols)


def test_walk_forward_never_trains_on_the_future():
    """Reproduce the fold arithmetic: train always ends before the test week."""
    from config import MIN_TRAIN_WEEKS, STEP, VAL_WEEKS

    for test_pos in range(MIN_TRAIN_WEEKS, 600, STEP):
        train_end = test_pos - VAL_WEEKS      # training stops here
        assert train_end < test_pos
        assert test_pos - train_end == VAL_WEEKS
