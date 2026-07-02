"""Recent form, streak, and momentum features."""

from collections import defaultdict

import pandas as pd

from src.config import MOMENTUM_DAYS_LONG, MOMENTUM_DAYS_SHORT, ROUND_DEPTH


def compute_momentum_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum features for each match.

    Features use only data BEFORE the current match (no leakage).

    Returns DataFrame with columns:
        match_idx, w_win_streak, l_win_streak,
        w_wins_30d, l_wins_30d, w_matches_30d, l_matches_30d,
        w_best_result_90d, l_best_result_90d,
        w_titles_12m, l_titles_12m,
        w_grass_titles_24m, l_grass_titles_24m,
        w_days_since_last, l_days_since_last
    """
    # Track per-player state
    streaks: dict[int, int] = defaultdict(int)             # current win streak
    match_dates: dict[int, list] = defaultdict(list)       # (date, won, round, surface, tourney_id)
    last_match_date: dict[int, pd.Timestamp] = {}

    rows: list[dict] = []

    for idx, match in matches.iterrows():
        w_id = str(match["winner_id"])
        l_id = str(match["loser_id"])
        date = match["tourney_date"]
        rnd = str(match.get("round", ""))
        surface = str(match.get("surface", "")).capitalize()
        tourney = match.get("tourney_id", "")

        round_depth = ROUND_DEPTH.get(rnd, 1)

        # ── Compute features from history (before this match) ──

        def _player_features(pid: int) -> dict:
            hist = match_dates[pid]
            feat: dict = {}

            # Win streak
            feat["win_streak"] = streaks[pid]

            # Wins / matches in last N days
            if isinstance(date, pd.Timestamp):
                cutoff_short = date - pd.Timedelta(days=MOMENTUM_DAYS_SHORT)
                cutoff_long = date - pd.Timedelta(days=MOMENTUM_DAYS_LONG)

                recent_short = [h for h in hist if h[0] >= cutoff_short]
                feat["wins_30d"] = sum(1 for h in recent_short if h[1])
                feat["matches_30d"] = len(recent_short)

                recent_long = [h for h in hist if h[0] >= cutoff_long]
                feat["best_result_90d"] = (
                    max((h[2] for h in recent_long), default=0)
                )

                # Titles in last 12 months (won final)
                cutoff_12m = date - pd.Timedelta(days=365)
                hist_12m = [h for h in hist if h[0] >= cutoff_12m]
                feat["titles_12m"] = sum(
                    1 for h in hist_12m if h[1] and h[2] == 7  # F = 7
                )

                # Grass titles in last 24 months
                cutoff_24m = date - pd.Timedelta(days=730)
                hist_24m = [h for h in hist if h[0] >= cutoff_24m]
                feat["grass_titles_24m"] = sum(
                    1 for h in hist_24m
                    if h[1] and h[2] == 7 and h[3] == "Grass"
                )

                # Days since last match
                if pid in last_match_date:
                    feat["days_since_last"] = (
                        date - last_match_date[pid]
                    ).days
                else:
                    feat["days_since_last"] = 90  # default for first match
            else:
                feat.update({
                    "wins_30d": 0, "matches_30d": 0, "best_result_90d": 0,
                    "titles_12m": 0, "grass_titles_24m": 0,
                    "days_since_last": 90,
                })

            return feat

        w_feat = _player_features(w_id)
        l_feat = _player_features(l_id)

        rows.append({
            "match_idx": idx,
            **{f"w_{k}": v for k, v in w_feat.items()},
            **{f"l_{k}": v for k, v in l_feat.items()},
        })

        # ── Update state after this match ──
        streaks[w_id] = streaks[w_id] + 1
        streaks[l_id] = 0

        match_dates[w_id].append((date, True, round_depth, surface, tourney))
        match_dates[l_id].append((date, False, round_depth, surface, tourney))

        last_match_date[w_id] = date
        last_match_date[l_id] = date

    return pd.DataFrame(rows)
