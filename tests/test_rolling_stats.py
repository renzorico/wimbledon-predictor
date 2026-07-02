import unittest

import pandas as pd

from src.features.rolling_stats import compute_rolling_stats


def _make_match(
    tourney_date: str,
    winner_id: str,
    loser_id: str,
    w_serve_pct: float,
    l_serve_pct: float,
) -> dict:
    w_1st_in = 40
    l_1st_in = 40
    return {
        "tourney_date": pd.Timestamp(tourney_date),
        "surface": "Grass",
        "winner_id": winner_id,
        "loser_id": loser_id,
        "w_serve_pct": w_serve_pct,
        "w_1st_in_pct": 0.60,
        "w_1st_won_pct": 0.72,
        "w_2nd_won_pct": 0.50,
        "w_return_pct": 1 - l_serve_pct,
        "w_bp_saved_pct": 0.65,
        "w_ace_rate": 0.40,
        "w_df_rate": 0.08,
        "l_serve_pct": l_serve_pct,
        "l_1st_in_pct": 0.58,
        "l_1st_won_pct": 0.68,
        "l_2nd_won_pct": 0.46,
        "l_return_pct": 1 - w_serve_pct,
        "l_bp_saved_pct": 0.60,
        "l_ace_rate": 0.30,
        "l_df_rate": 0.10,
    }


class RollingStatsLeakageTest(unittest.TestCase):
    def test_rolling_features_use_only_prior_matches(self) -> None:
        matches = pd.DataFrame(
            [
                _make_match("2026-06-01", "A", "B", 0.50, 0.55),
                _make_match("2026-06-08", "A", "C", 0.60, 0.54),
                _make_match("2026-06-15", "A", "D", 0.70, 0.53),
                _make_match("2026-06-22", "A", "E", 0.95, 0.52),
            ]
        )

        rolling = compute_rolling_stats(matches)
        player_a = rolling[rolling["player_id"] == "A"].reset_index(drop=True)

        self.assertTrue(pd.isna(player_a.loc[0, "roll_serve_pct_10"]))
        self.assertTrue(pd.isna(player_a.loc[1, "roll_serve_pct_10"]))
        self.assertTrue(pd.isna(player_a.loc[2, "roll_serve_pct_10"]))

        self.assertAlmostEqual(
            player_a.loc[3, "roll_serve_pct_10"],
            (0.50 + 0.60 + 0.70) / 3,
            places=6,
        )
        self.assertAlmostEqual(player_a.loc[3, "roll_win_rate_10"], 1.0, places=6)
        self.assertAlmostEqual(
            player_a.loc[3, "grass_roll_serve_pct_5"],
            (0.50 + 0.60 + 0.70) / 3,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
