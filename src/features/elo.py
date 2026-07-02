"""Standard + surface-weighted Elo rating system."""

from dataclasses import dataclass, field

import pandas as pd

from src.config import ELO_INITIAL, ELO_K_FACTOR, WELO_WEIGHTS


@dataclass
class EloRecord:
    """Elo ratings for a single player across surfaces."""
    overall: float = ELO_INITIAL
    grass: float = ELO_INITIAL
    hard: float = ELO_INITIAL
    clay: float = ELO_INITIAL
    matches_played: int = 0
    grass_matches: int = 0

    @property
    def welo(self) -> float:
        """Surface-weighted Elo optimized for grass prediction."""
        return (
            WELO_WEIGHTS["grass"] * self.grass
            + WELO_WEIGHTS["overall"] * self.overall
            + WELO_WEIGHTS["hard"] * self.hard
        )


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given their Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _k_factor(matches_played: int) -> float:
    """Adaptive K-factor: higher for new players, decaying with experience."""
    if matches_played < 30:
        return ELO_K_FACTOR * 1.5
    if matches_played < 100:
        return ELO_K_FACTOR
    return ELO_K_FACTOR * 0.75


def compute_elo_ratings(
    matches: pd.DataFrame,
) -> dict[str, EloRecord]:
    """Compute Elo ratings chronologically for all players.

    Args:
        matches: Cleaned match DataFrame sorted by tourney_date.

    Returns:
        Dict mapping player_id to their current EloRecord.
    """
    ratings: dict[int, EloRecord] = {}

    for _, row in matches.iterrows():
        w_id = str(row["winner_id"])
        l_id = str(row["loser_id"])
        surface = str(row.get("surface", "")).capitalize()

        # Initialize if new
        if w_id not in ratings:
            ratings[w_id] = EloRecord()
        if l_id not in ratings:
            ratings[l_id] = EloRecord()

        w_rec = ratings[w_id]
        l_rec = ratings[l_id]

        # Overall Elo update
        exp_w = _expected_score(w_rec.overall, l_rec.overall)
        k_w = _k_factor(w_rec.matches_played)
        k_l = _k_factor(l_rec.matches_played)
        w_rec.overall += k_w * (1 - exp_w)
        l_rec.overall += k_l * (0 - (1 - exp_w))

        # Surface-specific Elo update
        if surface == "Grass":
            exp_s = _expected_score(w_rec.grass, l_rec.grass)
            w_rec.grass += k_w * (1 - exp_s)
            l_rec.grass += k_l * (0 - (1 - exp_s))
            w_rec.grass_matches += 1
            l_rec.grass_matches += 1
        elif surface == "Hard":
            exp_s = _expected_score(w_rec.hard, l_rec.hard)
            w_rec.hard += k_w * (1 - exp_s)
            l_rec.hard += k_l * (0 - (1 - exp_s))
        elif surface == "Clay":
            exp_s = _expected_score(w_rec.clay, l_rec.clay)
            w_rec.clay += k_w * (1 - exp_s)
            l_rec.clay += k_l * (0 - (1 - exp_s))

        w_rec.matches_played += 1
        l_rec.matches_played += 1

    return ratings


def compute_elo_history(
    matches: pd.DataFrame,
) -> tuple[dict[int, EloRecord], pd.DataFrame]:
    """Compute Elo ratings and return history for each match.

    Returns:
        Tuple of (final ratings dict, DataFrame with Elo columns appended).
    """
    ratings: dict[str, EloRecord] = {}
    elo_rows: list[dict] = []

    for idx, row in matches.iterrows():
        w_id = str(row["winner_id"])
        l_id = str(row["loser_id"])
        surface = str(row.get("surface", "")).capitalize()

        if w_id not in ratings:
            ratings[w_id] = EloRecord()
        if l_id not in ratings:
            ratings[l_id] = EloRecord()

        w_rec = ratings[w_id]
        l_rec = ratings[l_id]

        # Snapshot BEFORE update (what the model sees at prediction time)
        elo_rows.append({
            "match_idx": idx,
            "w_elo": w_rec.overall,
            "l_elo": l_rec.overall,
            "w_elo_grass": w_rec.grass,
            "l_elo_grass": l_rec.grass,
            "w_elo_hard": w_rec.hard,
            "l_elo_hard": l_rec.hard,
            "w_welo": w_rec.welo,
            "l_welo": l_rec.welo,
        })

        # Update ratings
        exp_w = _expected_score(w_rec.overall, l_rec.overall)
        k_w = _k_factor(w_rec.matches_played)
        k_l = _k_factor(l_rec.matches_played)
        w_rec.overall += k_w * (1 - exp_w)
        l_rec.overall += k_l * (0 - (1 - exp_w))

        if surface == "Grass":
            exp_s = _expected_score(w_rec.grass, l_rec.grass)
            w_rec.grass += k_w * (1 - exp_s)
            l_rec.grass += k_l * (0 - (1 - exp_s))
            w_rec.grass_matches += 1
            l_rec.grass_matches += 1
        elif surface == "Hard":
            exp_s = _expected_score(w_rec.hard, l_rec.hard)
            w_rec.hard += k_w * (1 - exp_s)
            l_rec.hard += k_l * (0 - (1 - exp_s))
        elif surface == "Clay":
            exp_s = _expected_score(w_rec.clay, l_rec.clay)
            w_rec.clay += k_w * (1 - exp_s)
            l_rec.clay += k_l * (0 - (1 - exp_s))

        w_rec.matches_played += 1
        l_rec.matches_played += 1

    elo_df = pd.DataFrame(elo_rows)
    return ratings, elo_df
