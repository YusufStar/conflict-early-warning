"""
Feature engineering: lags, trend, month/country dummies, time-based split.
No leakage: lags/trend use only past data per country.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple

LAG_MONTHS_DEFAULT = [1, 2, 3, 6, 12]
TREND_WINDOW = 6


def add_lags(df: pd.DataFrame, lag_months: list[int] | None = None) -> pd.DataFrame:
    """Add event count lags per country. Requires sorted by country, period_index."""
    lag_months = lag_months or LAG_MONTHS_DEFAULT
    out = df.copy()
    for L in lag_months:
        out[f"events_lag{L}"] = out.groupby("country")["events"].shift(L)
    return out


def add_trend(df: pd.DataFrame, window: int = TREND_WINDOW) -> pd.DataFrame:
    """Add simple linear trend (slope) over last `window` months per country."""
    out = df.copy()

    def slope(x: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        t = np.arange(n, dtype=float)
        return float(np.polyfit(t, x.astype(float), 1)[0])

    out["events_trend"] = (
        out.groupby("country")["events"]
        .transform(lambda s: s.rolling(window, min_periods=1).apply(slope, raw=False))
    )
    return out


def add_rolling_stats(df: pd.DataFrame, windows: list[int] = [3, 6]) -> pd.DataFrame:
    """Optional: rolling mean and std of events per country."""
    out = df.copy()
    for w in windows:
        out[f"events_roll_mean_{w}"] = out.groupby("country")["events"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"events_roll_std_{w}"] = out.groupby("country")["events"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
    return out


def add_month_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add month (1-12) one-hot; column names month_1 .. month_12."""
    out = df.copy()
    for m in range(1, 13):
        out[f"month_{m}"] = (out["month"] == m).astype(int)
    return out


def add_country_encoding(
    df: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, OneHotEncoder | None]:
    """
    Add country one-hot. If encoder is None and fit=True, fit new one.
    Returns (df with country_0, country_1, ...), encoder.
    """
    out = df.copy()
    countries = out["country"].astype(str).values.reshape(-1, 1)
    if encoder is None:
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    if fit:
        enc = encoder.fit(countries)
    else:
        enc = encoder
    X = enc.transform(countries)
    names = [f"country_{i}" for i in range(X.shape[1])]
    country_df = pd.DataFrame(X, columns=names, index=out.index)
    out = pd.concat([out, country_df], axis=1)
    return out, enc


def build_features(
    panel: pd.DataFrame,
    lag_months: list[int] | None = None,
    add_rolling: bool = True,
    add_month_dum: bool = True,
    country_encoder: OneHotEncoder | None = None,
    fit_country_encoder: bool = True,
) -> Tuple[pd.DataFrame, OneHotEncoder | None]:
    """
    Build full feature set. Drops rows with NaN in lags (first max(lag_months) per country).
    Returns (df with feature columns), country_encoder.
    """
    df = add_lags(panel, lag_months)
    df = add_trend(df)
    if add_rolling:
        df = add_rolling_stats(df)
    if add_month_dum:
        df = add_month_dummies(df)
    df, enc = add_country_encoding(df, encoder=country_encoder, fit=fit_country_encoder)

    lag_cols = [c for c in df.columns if c.startswith("events_lag")]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    # Fill remaining NaN (e.g. rolling std) with 0 for model compatibility
    feat_cols = [c for c in df.columns if c not in {"country", "year", "month", "period_index", "events", "target_binary", "target_events", "target_events_next"}]
    df[feat_cols] = df[feat_cols].fillna(0)
    return df, enc


def time_based_split(
    df: pd.DataFrame,
    test_months: int = 12,
    val_months: int = 12,
    period_col: str = "period_index",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time: last test_months -> test, previous val_months -> val, rest -> train.
    """
    periods = sorted(df[period_col].unique())
    n = len(periods)
    if test_months + val_months >= n:
        test_months = max(1, n // 3)
        val_months = max(1, (n - test_months) // 2)
    test_cut = periods[-test_months]
    val_cut = periods[-(test_months + val_months)]
    train_df = df[df[period_col] < val_cut]
    val_df = df[(df[period_col] >= val_cut) & (df[period_col] < test_cut)]
    test_df = df[df[period_col] >= test_cut]
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Column names that are features (exclude id and target)."""
    exclude = {"country", "year", "month", "period_index", "events", "target_binary", "target_events", "target_events_next"}
    return [c for c in df.columns if c not in exclude]
