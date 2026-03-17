"""Load COW alliance and dyadic MID; aggregate to country-year."""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_ccode_to_name(alliance_member_path: str | Path) -> pd.Series:
    """From alliance_v4.1_by_member_yearly get unique ccode -> state_name."""
    path = Path(alliance_member_path)
    df = pd.read_csv(path)
    # ccode, state_name
    if "ccode" not in df.columns or "state_name" not in df.columns:
        return pd.Series(dtype=object)
    m = df[["ccode", "state_name"]].drop_duplicates().set_index("ccode")["state_name"]
    return m


def load_alliance_count_by_year(path: str | Path) -> pd.DataFrame:
    """
    Load alliance_v4.1_by_member_yearly; return (country, year) -> n_alliances.
    country = state_name.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df["country"] = df["state_name"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    cnt = df.groupby(["country", "year"]).size().reset_index(name="alliance_count")
    return cnt


def load_mid_count_by_year(
    dyadic_mid_path: str | Path,
    ccode_to_name: Optional[pd.Series] = None,
    alliance_member_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load dyadic_mid; count MIDs per (country, year). Country = state name from ccode.
    If ccode_to_name is None and alliance_member_path is set, build it from alliance.
    """
    path = Path(dyadic_mid_path)
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    if ccode_to_name is None and alliance_member_path is not None:
        ccode_to_name = load_ccode_to_name(alliance_member_path)
    if ccode_to_name is None or ccode_to_name.empty:
        return pd.DataFrame(columns=["country", "year", "mid_count"])

    def map_ccode(c):
        return ccode_to_name.get(int(c), None) if pd.notna(c) else None

    df["country_a"] = df["statea"].map(map_ccode)
    df["country_b"] = df["stateb"].map(map_ccode)
    rows = []
    for _, r in df.iterrows():
        y = r["year"]
        for c in (r["country_a"], r["country_b"]):
            if pd.notna(c) and str(c).strip():
                rows.append({"country": str(c).strip(), "year": y})
    if not rows:
        return pd.DataFrame(columns=["country", "year", "mid_count"])
    out = pd.DataFrame(rows).groupby(["country", "year"]).size().reset_index(name="mid_count")
    return out


def load_mid_history(
    dyadic_mid_path: str | Path,
    alliance_member_path: str | Path,
) -> dict[str, list[dict]]:
    """
    Load dyadic MID; for each country return list of {year, opponent}.
    Keys are state names (from alliance). Used for country detail view.
    """
    path = Path(dyadic_mid_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    ccode_to_name = load_ccode_to_name(alliance_member_path) if Path(alliance_member_path).exists() else pd.Series(dtype=object)
    if ccode_to_name is None or ccode_to_name.empty:
        return {}

    def map_ccode(c):
        try:
            return ccode_to_name.get(int(c), None) if pd.notna(c) else None
        except (ValueError, TypeError):
            return None

    df["name_a"] = df["statea"].map(map_ccode)
    df["name_b"] = df["stateb"].map(map_ccode)
    out = {}
    for _, r in df.iterrows():
        a, b = str(r["name_a"] or "").strip(), str(r["name_b"] or "").strip()
        y = int(r["year"])
        if a:
            out.setdefault(a, []).append({"year": y, "opponent": b or "Unknown"})
        if b:
            out.setdefault(b, []).append({"year": y, "opponent": a or "Unknown"})
    for k in out:
        out[k] = sorted(out[k], key=lambda x: (-x["year"], x["opponent"]))[:200]
    return out
