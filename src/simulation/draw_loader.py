"""Load the 2026 Wimbledon draw and map to TML player IDs."""

import hashlib

import pandas as pd

from src.config import PROCESSED_DIR, WIMBLEDON_2026_DIR
from src.simulation.bracket import Match, Player, WimbledonBracket


def _build_name_to_id(matches: pd.DataFrame) -> dict[str, str]:
    """Build a name -> player_id lookup from match history."""
    lookup: dict[str, str] = {}
    for _, row in matches.iterrows():
        lookup[str(row["winner_name"])] = str(row["winner_id"])
        lookup[str(row["loser_name"])] = str(row["loser_id"])
    return lookup


def _fuzzy_lookup(name: str, lookup: dict[str, str]) -> str | None:
    """Find player_id by exact or last-name match."""
    # Exact match
    if name in lookup:
        return lookup[name]
    # Last-name-only match
    last = name.strip().split()[-1]
    matches = [pid for n, pid in lookup.items() if n.split()[-1].lower() == last.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def _stable_new_player_id(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"NEW_{digest}"


def load_wimbledon_2026_draw() -> WimbledonBracket:
    """Load the 128-player draw from data/wimbledon_2026/draw.csv.

    Locks in known R1 results from the r1_result column.
    """
    draw_path = WIMBLEDON_2026_DIR / "draw.csv"
    if not draw_path.exists():
        raise FileNotFoundError(f"{draw_path} not found.")

    # Load draw
    draw = pd.read_csv(draw_path)
    draw["seed"] = pd.to_numeric(draw["seed"], errors="coerce")
    draw = draw.sort_values("position").reset_index(drop=True)

    if len(draw) != 128:
        raise ValueError(f"draw.csv has {len(draw)} rows, expected 128")

    # Build name -> ID lookup from match history
    matches = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    name_to_id = _build_name_to_id(matches)

    # Build Player list in draw order
    players: list[Player] = []
    for _, row in draw.iterrows():
        name = str(row["name"]).strip()
        seed = int(row["seed"]) if pd.notna(row["seed"]) else None
        nation = str(row.get("nation", "")).strip()

        pid = _fuzzy_lookup(name, name_to_id)
        if pid is None:
            pid = _stable_new_player_id(name)

        players.append(Player(
            player_id=pid,
            name=name,
            seed=seed,
            nation=nation,
        ))

    bracket = WimbledonBracket(players)

    # Lock known R1 results
    known = draw[draw["r1_result"].notna() & (draw["r1_result"] == "W")]
    for _, row in known.iterrows():
        pos = int(row["position"])
        # Match number = ceil(position / 2)
        match_num = (pos + 1) // 2
        match_id = f"R1_M{match_num}"
        if match_id in bracket.matches:
            match = bracket.matches[match_id]
            # Determine which slot the winner is in
            winner_slot = "a" if pos % 2 == 1 else "b"
            winner = match.player_a if winner_slot == "a" else match.player_b
            if winner:
                bracket.lock_result(match_id, winner)

    return bracket


def get_bracket_summary(bracket: WimbledonBracket) -> dict:
    """Return summary stats about the bracket state."""
    locked = sum(1 for m in bracket.matches.values() if m.locked)
    total_r1 = 64
    return {
        "r1_complete": locked,
        "r1_remaining": total_r1 - locked,
        "r1_pct": f"{locked / total_r1:.0%}",
    }
