"""Load VIEWS Fatalities Data (country-month)."""
import pandas as pd
from pathlib import Path


def load_views_fatalities(path: str | Path) -> pd.DataFrame:
    """
    Load VIEWS Fatalities CSV. Expects: country_id, month_id, country, gwcode, isoab, year, month,
    main_mean_ln, main_mean, main_dich.
    Returns df with country, year, month, views_main_mean, views_main_dich (and optionally main_mean_ln).
    """
    path = Path(path)
    df = pd.read_csv(path)
    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").dropna().astype(int)
    df = df.dropna(subset=["country", "year", "month"])
    out = df[["country", "year", "month"]].copy()
    out["views_main_mean"] = pd.to_numeric(df["main_mean"], errors="coerce")
    out["views_main_dich"] = pd.to_numeric(df["main_dich"], errors="coerce")
    if "main_mean_ln" in df.columns:
        out["views_main_mean_ln"] = pd.to_numeric(df["main_mean_ln"], errors="coerce")
    return out
