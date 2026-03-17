"""Merge VIEWS, alliance, MID into base panel."""
import pandas as pd
from pathlib import Path
from typing import Optional

from .panel import load_and_build_panel
from .views import load_views_fatalities
from .alliances_mids import load_alliance_count_by_year, load_mid_count_by_year


def build_enriched_panel(
    political_violence_path: str | Path,
    views_path: Optional[str | Path] = None,
    alliance_path: Optional[str | Path] = None,
    dyadic_mid_path: Optional[str | Path] = None,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """
    Build panel from Political Violence Excel, then merge VIEWS fatalities (country-month),
    alliance count (country-year), MID count (country-year). Missing values filled with 0.
    """
    panel = load_and_build_panel(political_violence_path, fill_missing=fill_missing)

    if views_path and Path(views_path).exists():
        views = load_views_fatalities(views_path)
        panel = panel.merge(
            views,
            on=["country", "year", "month"],
            how="left",
        )
        for c in ["views_main_mean", "views_main_dich"]:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)
        if "views_main_mean_ln" in panel.columns:
            panel["views_main_mean_ln"] = pd.to_numeric(panel["views_main_mean_ln"], errors="coerce").fillna(0)

    if alliance_path and Path(alliance_path).exists():
        alliance = load_alliance_count_by_year(alliance_path)
        panel = panel.merge(
            alliance,
            on=["country", "year"],
            how="left",
        )
        panel["alliance_count"] = pd.to_numeric(panel.get("alliance_count", 0), errors="coerce").fillna(0)

    if dyadic_mid_path and Path(dyadic_mid_path).exists() and alliance_path and Path(alliance_path).exists():
        mid = load_mid_count_by_year(dyadic_mid_path, alliance_member_path=alliance_path)
        panel = panel.merge(
            mid,
            on=["country", "year"],
            how="left",
        )
        panel["mid_count"] = pd.to_numeric(panel.get("mid_count", 0), errors="coerce").fillna(0)

    return panel.sort_values(["country", "period_index"]).reset_index(drop=True)
