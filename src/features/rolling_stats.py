"""Rolling serve/return/win statistics per player."""

import pandas as pd

from src.config import ROLLING_WINDOW, ROLLING_WINDOW_GRASS


# Stats to track per match (from winner/loser perspective)
_PLAYER_STATS = [
    "serve_pct", "1st_in_pct", "1st_won_pct", "2nd_won_pct",
    "return_pct", "bp_saved_pct", "ace_rate", "df_rate",
]


def _unify_player_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches so each row is one player's stats from one match.

    Produces columns: player_id, tourney_date, surface, won (bool), and all
    stat columns without the w_/l_ prefix.
    """
    # Winner rows
    ordered = matches.reset_index(names="match_idx")

    w_cols = {
        "winner_id": "player_id",
        "tourney_date": "tourney_date",
        "surface": "surface",
    }
    w_stats = {f"w_{s}": s for s in _PLAYER_STATS}
    winners = ordered.rename(columns={**w_cols, **w_stats})[
        ["match_idx", "player_id", "tourney_date", "surface"] + _PLAYER_STATS
    ].copy()
    winners["won"] = True

    # Loser rows
    l_cols = {
        "loser_id": "player_id",
        "tourney_date": "tourney_date",
        "surface": "surface",
    }
    l_stats = {f"l_{s}": s for s in _PLAYER_STATS}
    losers = ordered.rename(columns={**l_cols, **l_stats})[
        ["match_idx", "player_id", "tourney_date", "surface"] + _PLAYER_STATS
    ].copy()
    losers["won"] = False

    unified = pd.concat([winners, losers], ignore_index=True)
    unified = unified.sort_values(
        ["player_id", "tourney_date", "match_idx"],
        kind="stable",
    ).reset_index(drop=True)
    return unified


def compute_rolling_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player rolling stats and return a lookup DataFrame.

    Returns DataFrame with: player_id, tourney_date, surface, and rolling
    columns like roll_{stat}_{window} for all-surface and grass-specific.
    """
    unified = _unify_player_stats(matches)

    # All-surface rolling averages
    grouped = unified.groupby("player_id")
    for stat in _PLAYER_STATS:
        unified[f"roll_{stat}_{ROLLING_WINDOW}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
        )

    # Win rate rolling
    unified[f"roll_win_rate_{ROLLING_WINDOW}"] = grouped["won"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )

    # Grass-specific rolling averages
    grass = unified[unified["surface"].str.lower() == "grass"].copy()
    grass_grouped = grass.groupby("player_id")
    for stat in _PLAYER_STATS:
        grass[f"grass_roll_{stat}_{ROLLING_WINDOW_GRASS}"] = (
            grass_grouped[stat].transform(
                lambda s: s.shift(1).rolling(ROLLING_WINDOW_GRASS, min_periods=2).mean()
            )
        )
    grass[f"grass_roll_win_rate_{ROLLING_WINDOW_GRASS}"] = (
        grass_grouped["won"].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW_GRASS, min_periods=2).mean()
        )
    )

    # Merge grass rolling stats back
    grass_cols = [c for c in grass.columns if c.startswith("grass_roll_")]
    grass_merge = grass[["player_id", "tourney_date"] + grass_cols]

    unified = unified.merge(
        grass_merge,
        on=["player_id", "tourney_date"],
        how="left",
    )

    return unified
