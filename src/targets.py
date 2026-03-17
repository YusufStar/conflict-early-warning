"""
Targets: binary high-risk (next month events > threshold) and next-month event count.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def next_period_events(df: pd.DataFrame, events_col: str = "events") -> pd.Series:
    """Next month event count per row (per country)."""
    return df.groupby("country")[events_col].shift(-1)


def add_target_regression(df: pd.DataFrame, events_col: str = "events") -> pd.DataFrame:
    """Add target_events = next month events. Drops last month per country (NaN target)."""
    out = df.copy()
    out["target_events"] = next_period_events(out, events_col)
    return out.dropna(subset=["target_events"]).reset_index(drop=True)


def add_target_binary(
    df: pd.DataFrame,
    events_col: str = "events",
    threshold: float | None = None,
    percentile: float = 75.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Add target_binary = 1 if next month events > threshold else 0.
    If threshold is None, use percentile of (current) event distribution.
    Returns (df with target_binary, threshold used). Drops last month per country.
    """
    out = df.copy()
    next_ev = next_period_events(out, events_col)
    out["target_events_next"] = next_ev
    if threshold is None:
        threshold = float(np.nanpercentile(out["events"].values, percentile))
    out["target_binary"] = (next_ev > threshold).astype(int)
    out = out.dropna(subset=["target_binary"]).reset_index(drop=True)
    return out, threshold


def add_targets(
    df: pd.DataFrame,
    target_type: str,
    events_col: str = "events",
    high_risk_percentile: float = 75.0,
    high_risk_threshold: float | None = None,
) -> Tuple[pd.DataFrame, float | None]:
    """
    target_type in ("binary", "regression").
    Returns (df with target column(s)), threshold used for binary or None.
    """
    if target_type == "regression":
        out = add_target_regression(df, events_col)
        return out, None
    if target_type == "binary":
        out, th = add_target_binary(
            df, events_col,
            threshold=high_risk_threshold,
            percentile=high_risk_percentile,
        )
        return out, th
    raise ValueError(f"target_type must be 'binary' or 'regression', got {target_type}")
