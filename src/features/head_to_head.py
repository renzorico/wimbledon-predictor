"""Head-to-head record computation."""

from collections import defaultdict

import pandas as pd

from src.config import H2H_RECENT_N


def compute_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute H2H features for each match in the dataset.

    For each match (winner_id vs loser_id), computes stats using only
    matches that occurred BEFORE this one (no leakage).

    Returns DataFrame with columns:
        match_idx, h2h_wins_w, h2h_wins_l, h2h_win_pct_w,
        h2h_grass_wins_w, h2h_grass_wins_l, h2h_grass_win_pct_w,
        h2h_recent_{N}_w (proportion of last N meetings won by winner)
    """
    # Track all meetings between each pair
    # Key: frozenset({id_a, id_b}), Value: list of (date, winner_id, surface)
    history: dict[frozenset, list[tuple]] = defaultdict(list)
    rows: list[dict] = []

    for idx, match in matches.iterrows():
        w_id = str(match["winner_id"])
        l_id = str(match["loser_id"])
        surface = str(match.get("surface", "")).capitalize()
        pair = frozenset({w_id, l_id})

        prev = history[pair]

        # All-surface H2H
        h2h_wins_w = sum(1 for _, wid, _ in prev if wid == w_id)
        h2h_wins_l = sum(1 for _, wid, _ in prev if wid == l_id)
        total = h2h_wins_w + h2h_wins_l
        h2h_win_pct_w = h2h_wins_w / total if total > 0 else 0.5

        # Grass H2H
        grass_prev = [(d, wid, s) for d, wid, s in prev if s == "Grass"]
        h2h_grass_w = sum(1 for _, wid, _ in grass_prev if wid == w_id)
        h2h_grass_l = sum(1 for _, wid, _ in grass_prev if wid == l_id)
        grass_total = h2h_grass_w + h2h_grass_l
        h2h_grass_pct_w = h2h_grass_w / grass_total if grass_total > 0 else 0.5

        # Recent N meetings
        recent = prev[-H2H_RECENT_N:] if len(prev) >= H2H_RECENT_N else prev
        if recent:
            recent_w = sum(1 for _, wid, _ in recent if wid == w_id) / len(recent)
        else:
            recent_w = 0.5

        rows.append({
            "match_idx": idx,
            "h2h_wins_w": h2h_wins_w,
            "h2h_wins_l": h2h_wins_l,
            "h2h_win_pct_w": h2h_win_pct_w,
            "h2h_grass_wins_w": h2h_grass_w,
            "h2h_grass_wins_l": h2h_grass_l,
            "h2h_grass_win_pct_w": h2h_grass_pct_w,
            f"h2h_recent_{H2H_RECENT_N}_w": recent_w,
            "h2h_total_meetings": total,
        })

        # Record this match
        history[pair].append((match.get("tourney_date"), w_id, surface))

    return pd.DataFrame(rows)
