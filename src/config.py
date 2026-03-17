"""Data paths and defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PATHS = {
    "political_violence": DATA_DIR / "Political Violence Events by Country Mar 2026.xlsx",
    "views_fatalities": DATA_DIR / "VIEWS Fatalities Data 2026.csv",
    "alliance_member_yearly": DATA_DIR / "version4.1_csv" / "alliance_v4.1_by_member_yearly.csv",
    "dyadic_mid": DATA_DIR / "dyadic_mid_4.03.csv",
    "atop_sscore": DATA_DIR / "Atop Sscore.csv",
}

def get_path(key: str) -> Path:
    return PATHS.get(key, DATA_DIR / key)
