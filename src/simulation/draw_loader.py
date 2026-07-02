"""Load the 2026 Wimbledon draw and map to TML player IDs."""

import hashlib

import pandas as pd

from src.config import PROCESSED_DIR, WIMBLEDON_2026_DIR
from src.simulation.bracket import Match, Player, WimbledonBracket

_ROUND_TO_BRACKET = {
    "R128": ("R1", 2),
    "R64": ("R2", 4),
    "R32": ("R3", 8),
    "R16": ("R4", 16),
    "QF": ("QF", 32),
    "SF": ("SF", 64),
    "F": ("F", 128),
}


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


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _build_name_to_position(draw: pd.DataFrame) -> dict[str, int]:
    exact = {
        _normalize_name(row["name"]): int(row["position"])
        for _, row in draw.iterrows()
    }
    last_name_counts: dict[str, int] = {}
    last_name_positions: dict[str, int] = {}
    for _, row in draw.iterrows():
        parts = _normalize_name(row["name"]).split()
        if not parts:
            continue
        last_name = parts[-1]
        last_name_counts[last_name] = last_name_counts.get(last_name, 0) + 1
        last_name_positions[last_name] = int(row["position"])

    lookup = dict(exact)
    for last_name, count in last_name_counts.items():
        if count == 1:
            lookup[last_name] = last_name_positions[last_name]
    return lookup


def _lookup_draw_position(name: str, lookup: dict[str, int]) -> int | None:
    normalized = _normalize_name(name)
    if normalized in lookup:
        return lookup[normalized]
    parts = normalized.split()
    if parts and parts[-1] in lookup:
        return lookup[parts[-1]]
    return None


def _lock_completed_wimbledon_matches(
    bracket: WimbledonBracket,
    draw: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    """Lock completed Wimbledon matches from processed match data."""
    name_to_position = _build_name_to_position(draw)
    wimbledon = matches[
        (matches["tourney_id"].astype(str) == "2026-540")
        | (
            (matches["tourney_name"].astype(str).str.lower() == "wimbledon")
            & (matches["tourney_date"].dt.year == 2026)
        )
    ].sort_values(["tourney_date", "match_num"], kind="stable")

    for _, row in wimbledon.iterrows():
        round_info = _ROUND_TO_BRACKET.get(str(row.get("round", "")))
        if round_info is None:
            continue

        winner_name = str(row["winner_name"])
        position = _lookup_draw_position(winner_name, name_to_position)
        if position is None:
            continue

        round_name, block_size = round_info
        match_num = ((position - 1) // block_size) + 1
        match_id = f"{round_name}_M{match_num}"
        if match_id not in bracket.matches:
            continue
        if bracket.matches[match_id].locked:
            continue

        winner = bracket.players[position - 1]
        bracket.lock_result(match_id, winner)


def _lock_draw_r1_results(bracket: WimbledonBracket, draw: pd.DataFrame) -> None:
    known = draw[draw["r1_result"].notna() & (draw["r1_result"] == "W")]
    for _, row in known.iterrows():
        position = int(row["position"])
        match_num = ((position - 1) // 2) + 1
        match_id = f"R1_M{match_num}"
        if match_id not in bracket.matches or bracket.matches[match_id].locked:
            continue
        bracket.lock_result(match_id, bracket.players[position - 1])


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

    _lock_completed_wimbledon_matches(bracket, draw, matches)
    _lock_draw_r1_results(bracket, draw)

    return bracket


def get_bracket_summary(bracket: WimbledonBracket) -> dict:
    """Return summary stats about the bracket state."""
    locked = sum(1 for m in bracket.matches.values() if m.locked)
    locked_r1 = sum(
        1 for m in bracket.matches.values()
        if m.round_name == "R1" and m.locked
    )
    total_r1 = 64
    return {
        "r1_complete": locked_r1,
        "r1_remaining": total_r1 - locked_r1,
        "r1_pct": f"{locked_r1 / total_r1:.0%}",
        "locked_total": locked,
    }
