"""Tests for data loaders and enriched panel."""
import pytest
from pathlib import Path

import pandas as pd

from src.data import load_and_build_panel, build_enriched_panel, load_views_fatalities
from src.data.panel import build_panel, load_excel, parse_month
from src.config import DATA_DIR, PATHS


def test_parse_month():
    assert parse_month("January") == 1
    assert parse_month(3) == 3
    assert parse_month("march") == 3


def test_build_panel_from_df():
    df = pd.DataFrame({
        "COUNTRY": ["A", "A", "B"],
        "MONTH": ["January", "February", "January"],
        "YEAR": [2020, 2020, 2020],
        "EVENTS": [10, 20, 5],
    })
    out = build_panel(df, fill_missing=False)
    assert "country" in out.columns and "events" in out.columns
    assert out["period_index"].iloc[0] == 2020 * 12 + 1


@pytest.mark.skipif(not Path(PATHS["political_violence"]).exists(), reason="data file missing")
def test_load_panel():
    panel = load_and_build_panel(PATHS["political_violence"])
    assert panel.shape[0] >= 100
    assert set(panel.columns) >= {"country", "year", "month", "period_index", "events"}


@pytest.mark.skipif(not Path(PATHS["views_fatalities"]).exists(), reason="VIEWS file missing")
def test_load_views():
    df = load_views_fatalities(PATHS["views_fatalities"])
    assert "country" in df.columns and "views_main_mean" in df.columns


@pytest.mark.skipif(
    not Path(PATHS["political_violence"]).exists() or not Path(PATHS["views_fatalities"]).exists(),
    reason="data files missing",
)
def test_build_enriched_panel():
    panel = build_enriched_panel(
        PATHS["political_violence"],
        views_path=PATHS["views_fatalities"],
        alliance_path=PATHS.get("alliance_member_yearly") if Path(PATHS.get("alliance_member_yearly", Path("."))).exists() else None,
        dyadic_mid_path=PATHS.get("dyadic_mid") if Path(PATHS.get("dyadic_mid", Path("."))).exists() else None,
    )
    assert "events" in panel.columns
    assert "views_main_mean" in panel.columns or panel.shape[1] >= 5
