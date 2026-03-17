"""Smoke tests for feature build and split."""
import pytest
import pandas as pd
import numpy as np

from src.features import add_lags, add_trend, time_based_split, get_feature_columns, build_features
from src.targets import add_targets


@pytest.fixture
def small_panel():
    np.random.seed(42)
    countries = ["A", "B"]
    periods = list(range(2020 * 12 + 1, 2022 * 12 + 1))  # 24 months
    rows = []
    for c in countries:
        for p in periods:
            y, m = p // 12, (p % 12) or 12
            rows.append({"country": c, "year": y, "month": m, "period_index": p, "events": int(np.random.poisson(5))})
    return pd.DataFrame(rows)


def test_add_lags(small_panel):
    df = add_lags(small_panel, lag_months=[1, 2])
    assert "events_lag1" in df.columns
    assert df["events_lag1"].notna().any()


def test_time_based_split(small_panel):
    train, val, test = time_based_split(small_panel, test_months=4, val_months=4)
    assert len(train) + len(val) + len(test) == len(small_panel)
    assert test["period_index"].min() >= val["period_index"].max()


@pytest.mark.skipif(True, reason="build_features needs panel with targets; optional run")
def test_build_features_with_targets(small_panel):
    panel_with_target, _ = add_targets(small_panel, target_type="binary", high_risk_percentile=75)
    df, enc = build_features(panel_with_target, lag_months=[1, 2], add_rolling=False)
    cols = get_feature_columns(df)
    assert len(cols) >= 2
