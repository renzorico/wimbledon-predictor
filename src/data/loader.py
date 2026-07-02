"""Load and concatenate Sackmann CSVs into DataFrames."""

import pandas as pd

from src.config import MATCH_YEARS, RAW_DIR


def load_matches(
    years: range = MATCH_YEARS,
    surfaces: list[str] | None = None,
) -> pd.DataFrame:
    """Load match CSVs, concatenate, optionally filter by surface."""
    frames: list[pd.DataFrame] = []
    for year in years:
        path = RAW_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No match files found in {RAW_DIR}. Run download.py first."
        )

    matches = pd.concat(frames, ignore_index=True)

    # Parse date
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    if surfaces:
        surfaces_lower = [s.lower() for s in surfaces]
        matches = matches[
            matches["surface"].str.lower().isin(surfaces_lower)
        ].reset_index(drop=True)

    return matches


def load_players() -> pd.DataFrame:
    """Load player metadata (height, hand, DOB, country)."""
    path = RAW_DIR / "atp_players.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download.py first.")
    players = pd.read_csv(path, low_memory=False)
    # Parse DOB
    players["dob"] = pd.to_datetime(
        players["dob"], format="%Y%m%d", errors="coerce"
    )
    return players


def load_rankings() -> pd.DataFrame:
    """Load current ATP rankings snapshot."""
    path = RAW_DIR / "atp_rankings_current.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download.py first.")
    return pd.read_csv(path, low_memory=False)
