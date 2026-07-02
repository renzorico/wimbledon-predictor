"""Orchestrate all feature modules into a match-level feature matrix."""

import pandas as pd

from src.config import PROCESSED_DIR, ROLLING_WINDOW, ROLLING_WINDOW_GRASS
from src.features.elo import compute_elo_history
from src.features.head_to_head import compute_h2h_features
from src.features.momentum import compute_momentum_features
from src.features.rolling_stats import compute_rolling_stats


# Delta feature pairs: (winner_col, loser_col) -> delta_name
_DELTA_PAIRS = [
    # Elo
    ("w_elo", "l_elo", "elo_delta"),
    ("w_welo", "l_welo", "welo_delta"),
    ("w_elo_grass", "l_elo_grass", "elo_grass_delta"),
    ("w_elo_hard", "l_elo_hard", "elo_hard_delta"),
    # Ranking
    ("winner_rank", "loser_rank", "rank_delta"),
    ("winner_rank_points", "loser_rank_points", "rank_points_delta"),
    # Physical
    ("winner_age", "loser_age", "age_delta"),
    ("winner_ht", "loser_ht", "height_delta"),
    # Momentum
    ("w_win_streak", "l_win_streak", "win_streak_delta"),
    ("w_wins_30d", "l_wins_30d", "wins_30d_delta"),
    ("w_matches_30d", "l_matches_30d", "matches_30d_delta"),
    ("w_best_result_90d", "l_best_result_90d", "best_result_90d_delta"),
    ("w_titles_12m", "l_titles_12m", "titles_12m_delta"),
    ("w_grass_titles_24m", "l_grass_titles_24m", "grass_titles_24m_delta"),
    ("w_days_since_last", "l_days_since_last", "days_since_last_delta"),
]

# Rolling stat deltas
_W = ROLLING_WINDOW
_G = ROLLING_WINDOW_GRASS
_ROLLING_STATS = [
    "serve_pct", "1st_in_pct", "1st_won_pct", "2nd_won_pct",
    "return_pct", "bp_saved_pct", "ace_rate", "df_rate",
]


def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature matrix from cleaned matches.

    All features are deltas (winner - loser perspective). The target column
    'y' is always 1 (winner won). During training, rows are randomly flipped
    so the model learns symmetry.

    Args:
        matches: Cleaned match DataFrame sorted by tourney_date.

    Returns:
        DataFrame with feature columns + metadata (ids, date, surface).
    """
    print("  computing Elo ratings...")
    elo_ratings, elo_df = compute_elo_history(matches)

    print("  computing momentum features...")
    momentum_df = compute_momentum_features(matches)

    print("  computing head-to-head features...")
    h2h_df = compute_h2h_features(matches)

    print("  computing rolling stats...")
    rolling_df = compute_rolling_stats(matches)

    # ── Merge Elo into matches ──
    matches = matches.reset_index(drop=True)
    matches = matches.merge(elo_df, left_index=True, right_on="match_idx", how="left")
    matches = matches.merge(momentum_df, on="match_idx", how="left")
    matches = matches.merge(h2h_df, on="match_idx", how="left")

    # ── Merge rolling stats (winner side) ──
    w_rolling = rolling_df.rename(
        columns={
            "player_id": "winner_id",
            **{f"roll_{s}_{_W}": f"w_roll_{s}" for s in _ROLLING_STATS},
            f"roll_win_rate_{_W}": "w_roll_win_rate",
            **{f"grass_roll_{s}_{_G}": f"w_grass_roll_{s}" for s in _ROLLING_STATS},
            f"grass_roll_win_rate_{_G}": "w_grass_roll_win_rate",
        }
    )
    w_merge_cols = (
        ["winner_id", "tourney_date"]
        + [f"w_roll_{s}" for s in _ROLLING_STATS]
        + ["w_roll_win_rate"]
        + [f"w_grass_roll_{s}" for s in _ROLLING_STATS]
        + ["w_grass_roll_win_rate"]
    )
    w_available = [c for c in w_merge_cols if c in w_rolling.columns]
    matches = matches.merge(
        w_rolling[w_available].drop_duplicates(subset=["winner_id", "tourney_date"], keep="last"),
        on=["winner_id", "tourney_date"],
        how="left",
    )

    # ── Merge rolling stats (loser side) ──
    l_rolling = rolling_df.rename(
        columns={
            "player_id": "loser_id",
            **{f"roll_{s}_{_W}": f"l_roll_{s}" for s in _ROLLING_STATS},
            f"roll_win_rate_{_W}": "l_roll_win_rate",
            **{f"grass_roll_{s}_{_G}": f"l_grass_roll_{s}" for s in _ROLLING_STATS},
            f"grass_roll_win_rate_{_G}": "l_grass_roll_win_rate",
        }
    )
    l_merge_cols = (
        ["loser_id", "tourney_date"]
        + [f"l_roll_{s}" for s in _ROLLING_STATS]
        + ["l_roll_win_rate"]
        + [f"l_grass_roll_{s}" for s in _ROLLING_STATS]
        + ["l_grass_roll_win_rate"]
    )
    l_available = [c for c in l_merge_cols if c in l_rolling.columns]
    matches = matches.merge(
        l_rolling[l_available].drop_duplicates(subset=["loser_id", "tourney_date"], keep="last"),
        on=["loser_id", "tourney_date"],
        how="left",
    )

    # ── Compute delta features ──
    for w_col, l_col, delta_name in _DELTA_PAIRS:
        if w_col in matches.columns and l_col in matches.columns:
            # For rank, lower is better, so invert the delta
            if "rank" in delta_name and "points" not in delta_name:
                matches[delta_name] = matches[l_col] - matches[w_col]
            else:
                matches[delta_name] = matches[w_col] - matches[l_col]

    # Rolling stat deltas
    for stat in _ROLLING_STATS:
        w_c = f"w_roll_{stat}"
        l_c = f"l_roll_{stat}"
        if w_c in matches.columns and l_c in matches.columns:
            matches[f"roll_{stat}_delta"] = matches[w_c] - matches[l_c]

    # Grass rolling stat deltas
    for stat in _ROLLING_STATS:
        w_c = f"w_grass_roll_{stat}"
        l_c = f"l_grass_roll_{stat}"
        if w_c in matches.columns and l_c in matches.columns:
            matches[f"grass_roll_{stat}_delta"] = matches[w_c] - matches[l_c]

    # Win rate delta
    if "w_roll_win_rate" in matches.columns and "l_roll_win_rate" in matches.columns:
        matches["roll_win_rate_delta"] = matches["w_roll_win_rate"] - matches["l_roll_win_rate"]

    if "w_grass_roll_win_rate" in matches.columns and "l_grass_roll_win_rate" in matches.columns:
        matches["grass_roll_win_rate_delta"] = (
            matches["w_grass_roll_win_rate"] - matches["l_grass_roll_win_rate"]
        )

    # Surface binary
    matches["surface_grass"] = (matches["surface"].str.lower() == "grass").astype(int)

    # Target: always 1 from winner perspective
    matches["y"] = 1

    # ── Select output columns ──
    meta_cols = [
        "tourney_id", "tourney_name", "tourney_date", "surface", "round",
        "winner_id", "winner_name", "loser_id", "loser_name",
        "winner_seed", "loser_seed",
    ]
    delta_cols = [c for c in matches.columns if c.endswith("_delta")]
    h2h_cols = [c for c in matches.columns if c.startswith("h2h_")]
    extra = ["surface_grass", "y"]

    keep = meta_cols + delta_cols + h2h_cols + extra
    keep = [c for c in keep if c in matches.columns]

    return matches[keep]


def save_feature_matrix(df: pd.DataFrame) -> None:
    """Save the feature matrix to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "feature_matrix.parquet"
    df.to_parquet(path, index=False)
    n_features = len([c for c in df.columns if c.endswith("_delta") or c.startswith("h2h_")])
    print(f"saved {len(df):,} rows × {n_features} features to {path}")


if __name__ == "__main__":
    print("building feature matrix...")
    clean_path = PROCESSED_DIR / "matches_clean.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run src/data/clean.py first."
        )
    matches = pd.read_parquet(clean_path)
    features = build_feature_matrix(matches)
    save_feature_matrix(features)
