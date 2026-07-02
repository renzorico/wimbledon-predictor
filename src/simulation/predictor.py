"""Pre-match feature builder and model-backed match predictor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    MOMENTUM_DAYS_LONG,
    MOMENTUM_DAYS_SHORT,
    PROCESSED_DIR,
    ROUND_DEPTH,
    ROLLING_WINDOW,
    ROLLING_WINDOW_GRASS,
)
from src.features.elo import EloRecord, compute_elo_ratings
from src.models.elo_model import WeightedEloPredictor

CALIBRATED_MODEL_NAME = "XGBoost calibrated"
CALIBRATED_MODEL_WEIGHT = 0.35
CALIBRATED_TEMPERATURE = 2.75
CALIBRATED_MIN_PROBA = 0.10
CALIBRATED_MAX_PROBA = 0.90

_ROLLING_STATS = [
    "serve_pct",
    "1st_in_pct",
    "1st_won_pct",
    "2nd_won_pct",
    "return_pct",
    "bp_saved_pct",
    "ace_rate",
    "df_rate",
]


@dataclass
class PlayerState:
    """Latest known pre-match state for a player."""

    rank: float = np.nan
    rank_points: float = np.nan
    age: float = np.nan
    height: float = np.nan
    win_streak: float = 0.0
    wins_30d: float = 0.0
    matches_30d: float = 0.0
    best_result_90d: float = 0.0
    titles_12m: float = 0.0
    grass_titles_24m: float = 0.0
    days_since_last: float = 90.0
    roll_win_rate: float = np.nan
    grass_roll_win_rate: float = np.nan
    rolling_stats: dict[str, float] | None = None
    grass_rolling_stats: dict[str, float] | None = None


def _build_player_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape match data to one row per player appearance."""
    ordered = matches.reset_index(drop=True).copy()

    winner_rows = pd.DataFrame(
        {
            "player_id": ordered["winner_id"].astype(str),
            "player_name": ordered["winner_name"],
            "tourney_date": ordered["tourney_date"],
            "tourney_id": ordered["tourney_id"],
            "match_num": ordered["match_num"],
            "surface": ordered["surface"],
            "round": ordered["round"],
            "won": True,
            "rank": ordered["winner_rank"],
            "rank_points": ordered["winner_rank_points"],
            "age": ordered["winner_age"],
            "height": ordered["winner_ht"],
            **{stat: ordered[f"w_{stat}"] for stat in _ROLLING_STATS},
        }
    )

    loser_rows = pd.DataFrame(
        {
            "player_id": ordered["loser_id"].astype(str),
            "player_name": ordered["loser_name"],
            "tourney_date": ordered["tourney_date"],
            "tourney_id": ordered["tourney_id"],
            "match_num": ordered["match_num"],
            "surface": ordered["surface"],
            "round": ordered["round"],
            "won": False,
            "rank": ordered["loser_rank"],
            "rank_points": ordered["loser_rank_points"],
            "age": ordered["loser_age"],
            "height": ordered["loser_ht"],
            **{stat: ordered[f"l_{stat}"] for stat in _ROLLING_STATS},
        }
    )

    player_rows = pd.concat([winner_rows, loser_rows], ignore_index=True)
    return player_rows.sort_values(
        ["player_id", "tourney_date", "tourney_id", "match_num"],
        kind="stable",
    ).reset_index(drop=True)


def _latest_non_null(series: pd.Series) -> float:
    non_null = series.dropna()
    return non_null.iloc[-1] if not non_null.empty else np.nan


def _mean_recent(values: pd.Series, window: int, min_periods: int) -> float:
    if len(values) < min_periods:
        return np.nan
    return values.tail(window).mean()


def _safe_delta(value_a: float, value_b: float) -> float:
    if pd.isna(value_a) or pd.isna(value_b):
        return np.nan
    return float(value_a - value_b)


class MatchPredictor:
    """Builds pre-match features and scores player pairs with a trained model."""

    def __init__(
        self,
        matches: pd.DataFrame,
        model_name: str = "XGBoost",
        reference_date: str | pd.Timestamp | None = None,
    ) -> None:
        self.matches = matches.sort_values(
            ["tourney_date", "tourney_id", "match_num"],
            kind="stable",
        ).reset_index(drop=True)
        if reference_date is None:
            self.reference_date = self.matches["tourney_date"].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.Timestamp(reference_date)

        history = self.matches[self.matches["tourney_date"] < self.reference_date].copy()
        self.history = history.reset_index(drop=True)
        self.player_rows = _build_player_rows(self.history)
        self.elo_ratings = compute_elo_ratings(self.history)
        self.player_states = self._build_player_states()
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())

    @staticmethod
    def _load_model(model_name: str):
        if model_name == "Weighted Elo":
            return WeightedEloPredictor()
        if model_name == "Logistic Regression":
            return joblib.load(MODELS_DIR / "logistic_pipeline.pkl")
        if model_name in {"XGBoost", CALIBRATED_MODEL_NAME}:
            return joblib.load(MODELS_DIR / "xgb_pipeline.pkl")
        raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _temper_probability(probability: float) -> float:
        clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = np.log(clipped / (1 - clipped))
        tempered = 1.0 / (1.0 + np.exp(-logit / CALIBRATED_TEMPERATURE))
        return float(tempered)

    def _build_player_states(self) -> dict[str, PlayerState]:
        states: dict[str, PlayerState] = {}

        for player_id, rows in self.player_rows.groupby("player_id", sort=False):
            rows = rows.sort_values(
                ["tourney_date", "tourney_id", "match_num"],
                kind="stable",
            ).reset_index(drop=True)
            recent_30d = rows[rows["tourney_date"] >= self.reference_date - pd.Timedelta(days=MOMENTUM_DAYS_SHORT)]
            recent_90d = rows[rows["tourney_date"] >= self.reference_date - pd.Timedelta(days=MOMENTUM_DAYS_LONG)]
            recent_12m = rows[rows["tourney_date"] >= self.reference_date - pd.Timedelta(days=365)]
            recent_24m = rows[rows["tourney_date"] >= self.reference_date - pd.Timedelta(days=730)]
            grass_rows = rows[rows["surface"].str.lower() == "grass"]

            streak = 0
            for won in reversed(rows["won"].tolist()):
                if won:
                    streak += 1
                else:
                    break

            rolling_stats = {
                stat: _mean_recent(rows[stat], ROLLING_WINDOW, min_periods=3)
                for stat in _ROLLING_STATS
            }
            grass_rolling_stats = {
                stat: _mean_recent(grass_rows[stat], ROLLING_WINDOW_GRASS, min_periods=2)
                for stat in _ROLLING_STATS
            }

            states[str(player_id)] = PlayerState(
                rank=_latest_non_null(rows["rank"]),
                rank_points=_latest_non_null(rows["rank_points"]),
                age=_latest_non_null(rows["age"]),
                height=_latest_non_null(rows["height"]),
                win_streak=float(streak),
                wins_30d=float(recent_30d["won"].sum()),
                matches_30d=float(len(recent_30d)),
                best_result_90d=float(
                    recent_90d["round"].map(ROUND_DEPTH).fillna(1).max()
                    if not recent_90d.empty
                    else 0
                ),
                titles_12m=float(
                    ((recent_12m["won"]) & (recent_12m["round"] == "F")).sum()
                ),
                grass_titles_24m=float(
                    (
                        (recent_24m["won"])
                        & (recent_24m["round"] == "F")
                        & (recent_24m["surface"].str.lower() == "grass")
                    ).sum()
                ),
                days_since_last=float(
                    (self.reference_date - rows["tourney_date"].max()).days
                ),
                roll_win_rate=_mean_recent(rows["won"].astype(float), ROLLING_WINDOW, min_periods=3),
                grass_roll_win_rate=_mean_recent(
                    grass_rows["won"].astype(float),
                    ROLLING_WINDOW_GRASS,
                    min_periods=2,
                ),
                rolling_stats=rolling_stats,
                grass_rolling_stats=grass_rolling_stats,
            )

        return states

    def _get_player_state(self, player_id: str) -> PlayerState:
        return self.player_states.get(str(player_id), PlayerState(
            rolling_stats={stat: np.nan for stat in _ROLLING_STATS},
            grass_rolling_stats={stat: np.nan for stat in _ROLLING_STATS},
        ))

    def _get_elo_record(self, player_id: str) -> EloRecord:
        return self.elo_ratings.get(str(player_id), EloRecord())

    @lru_cache(maxsize=4096)
    def _h2h_features(self, player_a_id: str, player_b_id: str) -> dict[str, float]:
        pair_matches = self.history[
            (
                (self.history["winner_id"].astype(str) == player_a_id)
                & (self.history["loser_id"].astype(str) == player_b_id)
            )
            | (
                (self.history["winner_id"].astype(str) == player_b_id)
                & (self.history["loser_id"].astype(str) == player_a_id)
            )
        ].sort_values(["tourney_date", "tourney_id", "match_num"], kind="stable")

        total = len(pair_matches)
        wins_a = int((pair_matches["winner_id"].astype(str) == player_a_id).sum())
        wins_b = total - wins_a
        grass = pair_matches[pair_matches["surface"].str.lower() == "grass"]
        grass_wins_a = int((grass["winner_id"].astype(str) == player_a_id).sum())
        grass_wins_b = len(grass) - grass_wins_a
        recent = pair_matches.tail(3)
        recent_wins_a = int((recent["winner_id"].astype(str) == player_a_id).sum())

        return {
            "h2h_wins_w": wins_a,
            "h2h_wins_l": wins_b,
            "h2h_win_pct_w": wins_a / total if total else 0.5,
            "h2h_grass_wins_w": grass_wins_a,
            "h2h_grass_wins_l": grass_wins_b,
            "h2h_grass_win_pct_w": grass_wins_a / len(grass) if len(grass) else 0.5,
            "h2h_recent_3_w": recent_wins_a / len(recent) if len(recent) else 0.5,
            "h2h_total_meetings": total,
        }

    def build_feature_row(self, player_a_id: str, player_b_id: str) -> pd.DataFrame:
        player_a_id = str(player_a_id)
        player_b_id = str(player_b_id)

        state_a = self._get_player_state(player_a_id)
        state_b = self._get_player_state(player_b_id)
        elo_a = self._get_elo_record(player_a_id)
        elo_b = self._get_elo_record(player_b_id)

        row = {
            "elo_delta": elo_a.overall - elo_b.overall,
            "welo_delta": elo_a.welo - elo_b.welo,
            "elo_grass_delta": elo_a.grass - elo_b.grass,
            "elo_hard_delta": elo_a.hard - elo_b.hard,
            "rank_delta": state_b.rank - state_a.rank,
            "rank_points_delta": state_a.rank_points - state_b.rank_points,
            "age_delta": state_a.age - state_b.age,
            "height_delta": state_a.height - state_b.height,
            "win_streak_delta": state_a.win_streak - state_b.win_streak,
            "wins_30d_delta": state_a.wins_30d - state_b.wins_30d,
            "matches_30d_delta": state_a.matches_30d - state_b.matches_30d,
            "best_result_90d_delta": state_a.best_result_90d - state_b.best_result_90d,
            "titles_12m_delta": state_a.titles_12m - state_b.titles_12m,
            "grass_titles_24m_delta": state_a.grass_titles_24m - state_b.grass_titles_24m,
            "days_since_last_delta": state_a.days_since_last - state_b.days_since_last,
            "roll_win_rate_delta": state_a.roll_win_rate - state_b.roll_win_rate,
            "grass_roll_win_rate_delta": (
                state_a.grass_roll_win_rate - state_b.grass_roll_win_rate
            ),
            "surface_grass": 1,
        }

        for stat in _ROLLING_STATS:
            row[f"roll_{stat}_delta"] = _safe_delta(
                state_a.rolling_stats[stat],
                state_b.rolling_stats[stat],
            )
            row[f"grass_roll_{stat}_delta"] = _safe_delta(
                state_a.grass_rolling_stats[stat],
                state_b.grass_rolling_stats[stat],
            )

        row.update(self._h2h_features(player_a_id, player_b_id))

        feature_row = pd.DataFrame([row])
        return feature_row.reindex(columns=self.feature_columns)

    @lru_cache(maxsize=4096)
    def predict_proba(self, player_a_id: str, player_b_id: str) -> float:
        feature_row = self.build_feature_row(player_a_id, player_b_id).fillna(0)
        if self.model_name == "Weighted Elo":
            return float(self.model.predict_proba(feature_row)[0, 1])

        model_probability = float(self.model.predict_proba(feature_row)[0, 1])
        if self.model_name != CALIBRATED_MODEL_NAME:
            return model_probability

        elo_probability = float(WeightedEloPredictor().predict_proba(feature_row)[0, 1])
        tempered_model_probability = self._temper_probability(model_probability)
        blended = (
            CALIBRATED_MODEL_WEIGHT * tempered_model_probability
            + (1 - CALIBRATED_MODEL_WEIGHT) * elo_probability
        )
        return float(np.clip(blended, CALIBRATED_MIN_PROBA, CALIBRATED_MAX_PROBA))
