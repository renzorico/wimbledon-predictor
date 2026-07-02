"""Load the 2026 Wimbledon draw and map to Sackmann player IDs."""

import csv
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, WIMBLEDON_2026_DIR
from src.simulation.bracket import Player, WimbledonBracket


def _fuzzy_match_player(
    name: str,
    players_df: pd.DataFrame,
) -> int | None:
    """Find Sackmann player_id by fuzzy name matching."""
    # Try exact match first
    exact = players_df[
        players_df["name_full"].str.lower() == name.strip().lower()
    ]
    if len(exact) == 1:
        return int(exact.iloc[0]["player_id"])

    # Try last name match
    last_name = name.strip().split()[-1].lower()
    last_match = players_df[
        players_df["name_last"].str.lower() == last_name
    ]
    if len(last_match) == 1:
        return int(last_match.iloc[0]["player_id"])

    # Try first + last
    parts = name.strip().split()
    if len(parts) >= 2:
        first = parts[0].lower()
        last = parts[-1].lower()
        both = players_df[
            (players_df["name_first"].str.lower() == first)
            & (players_df["name_last"].str.lower() == last)
        ]
        if len(both) == 1:
            return int(both.iloc[0]["player_id"])

    return None


def load_players_lookup() -> pd.DataFrame:
    """Load Sackmann players CSV for ID matching."""
    path = RAW_DIR / "atp_players.csv"
    df = pd.read_csv(path, low_memory=False)
    # Create full name column
    df["name_full"] = df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    return df


def load_wimbledon_2026_draw() -> WimbledonBracket:
    """Load the 128-player draw from CSV.

    Expected CSV format: position, name, seed, nation
    (position 1-128 in draw order, top to bottom)
    """
    draw_path = WIMBLEDON_2026_DIR / "draw.csv"
    if not draw_path.exists():
        raise FileNotFoundError(
            f"{draw_path} not found. Create it with columns: "
            "position,name,seed,nation"
        )

    players_df = load_players_lookup()
    players: list[Player] = []

    with open(draw_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            seed = int(row["seed"]) if row.get("seed", "").strip() else None
            nation = row.get("nation", "").strip()

            player_id = _fuzzy_match_player(name, players_df)
            if player_id is None:
                # Use negative hash as fallback ID
                player_id = -abs(hash(name)) % 100000

            players.append(Player(
                player_id=player_id,
                name=name,
                seed=seed,
                nation=nation,
            ))

    if len(players) != 128:
        raise ValueError(f"Draw CSV has {len(players)} players, need 128")

    return WimbledonBracket(players)
