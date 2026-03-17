"""
Load Excel, parse month/year, build country-month panel.
"""
import pandas as pd
import numpy as np
from pathlib import Path

MONTH_STR_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def load_excel(path: str | Path) -> pd.DataFrame:
    """Load Excel; expect columns COUNTRY, MONTH, YEAR, EVENTS."""
    path = Path(path)
    df = pd.read_excel(path)
    cols = [c.upper().strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        u = c.upper().strip()
        if u in ("COUNTRY", "MONTH", "YEAR", "EVENTS"):
            rename[c] = u
    df = df.rename(columns=rename)
    for k in ("COUNTRY", "MONTH", "YEAR", "EVENTS"):
        if k not in df.columns:
            raise ValueError(f"Expected column {k} in {path}")
    return df


def parse_month(month_val) -> int:
    """Convert month name or number to 1-12."""
    if pd.isna(month_val):
        return np.nan
    if isinstance(month_val, (int, float)) and 1 <= month_val <= 12:
        return int(month_val)
    s = str(month_val).strip().lower()
    return MONTH_STR_TO_NUM.get(s, np.nan)


def build_panel(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """
    Build country-month panel: country, year, month, period_index, events.
    Sort by country, period_index. Optionally fill missing (country, period) with 0 events.
    """
    df = df.copy()
    df["month_num"] = df["MONTH"].map(parse_month)
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)
    df["year"] = df["YEAR"].astype(int)
    df["period_index"] = df["year"] * 12 + df["month_num"]
    df["country"] = df["COUNTRY"].astype(str).str.strip()
    df["events"] = pd.to_numeric(df["EVENTS"], errors="coerce").fillna(0).astype(int)

    out = df[["country", "year", "month_num", "period_index", "events"]].copy()
    out = out.rename(columns={"month_num": "month"})
    out = out.sort_values(["country", "period_index"]).reset_index(drop=True)

    if fill_missing:
        periods = out["period_index"].unique()
        countries = out["country"].unique()
        full = pd.MultiIndex.from_product(
            [countries, sorted(periods)],
            names=["country", "period_index"],
        )
        out = out.set_index(["country", "period_index"]).reindex(full, fill_value=0).reset_index()
        out["year"] = (out["period_index"] // 12).astype(int)
        out["month"] = (out["period_index"] % 12).replace(0, 12)
        out["events"] = out["events"].astype(int)
        out = out.sort_values(["country", "period_index"]).reset_index(drop=True)

    return out


def load_and_build_panel(path: str | Path, fill_missing: bool = True) -> pd.DataFrame:
    """Load Excel and return panel DataFrame."""
    df = load_excel(path)
    return build_panel(df, fill_missing=fill_missing)
