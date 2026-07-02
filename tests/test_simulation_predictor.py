import unittest

import pandas as pd

from src.simulation.draw_loader import _stable_new_player_id
from src.simulation.predictor import MatchPredictor


def _make_match(
    tourney_date: str,
    winner_id: str,
    winner_name: str,
    loser_id: str,
    loser_name: str,
    surface: str = "Grass",
    round_name: str = "R64",
    match_num: int = 1,
    winner_rank: int = 10,
    loser_rank: int = 20,
) -> dict:
    return {
        "tourney_date": pd.Timestamp(tourney_date),
        "tourney_id": f"T-{tourney_date}",
        "match_num": match_num,
        "surface": surface,
        "round": round_name,
        "winner_id": winner_id,
        "winner_name": winner_name,
        "loser_id": loser_id,
        "loser_name": loser_name,
        "winner_rank": winner_rank,
        "loser_rank": loser_rank,
        "winner_rank_points": 5000,
        "loser_rank_points": 3000,
        "winner_age": 24.0,
        "loser_age": 27.0,
        "winner_ht": 188.0,
        "loser_ht": 185.0,
        "w_serve_pct": 0.66,
        "w_1st_in_pct": 0.61,
        "w_1st_won_pct": 0.75,
        "w_2nd_won_pct": 0.53,
        "w_return_pct": 0.37,
        "w_bp_saved_pct": 0.64,
        "w_ace_rate": 0.45,
        "w_df_rate": 0.08,
        "l_serve_pct": 0.61,
        "l_1st_in_pct": 0.58,
        "l_1st_won_pct": 0.71,
        "l_2nd_won_pct": 0.48,
        "l_return_pct": 0.34,
        "l_bp_saved_pct": 0.59,
        "l_ace_rate": 0.32,
        "l_df_rate": 0.11,
    }


class SimulationPredictorTest(unittest.TestCase):
    def test_build_feature_row_matches_training_contract(self) -> None:
        matches = pd.DataFrame(
            [
                _make_match("2026-01-01", "A", "Alpha", "B", "Beta", match_num=1),
                _make_match("2026-01-08", "B", "Beta", "A", "Alpha", match_num=2),
                _make_match("2026-01-15", "A", "Alpha", "C", "Gamma", match_num=3),
                _make_match("2026-01-22", "D", "Delta", "B", "Beta", match_num=4),
            ]
        )

        predictor = MatchPredictor(
            matches=matches,
            model_name="Weighted Elo",
            reference_date="2026-02-01",
        )
        row = predictor.build_feature_row("A", "B")

        self.assertEqual(row.shape, (1, 42))
        self.assertEqual(row.loc[0, "h2h_total_meetings"], 2)
        self.assertEqual(row.loc[0, "h2h_wins_w"], 1)
        self.assertEqual(row.loc[0, "h2h_wins_l"], 1)
        self.assertAlmostEqual(row.loc[0, "h2h_win_pct_w"], 0.5, places=6)
        self.assertEqual(row.loc[0, "surface_grass"], 1)

        probability = predictor.predict_proba("A", "B")
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_stable_new_player_id_is_deterministic(self) -> None:
        self.assertEqual(
            _stable_new_player_id("Example Player"),
            _stable_new_player_id("Example Player"),
        )


if __name__ == "__main__":
    unittest.main()
