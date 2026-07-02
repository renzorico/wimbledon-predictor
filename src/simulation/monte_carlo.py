"""Monte Carlo tournament simulation."""

import copy
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.config import MONTE_CARLO_SIMS
from src.simulation.bracket import ROUNDS, WimbledonBracket


@dataclass
class SimulationResult:
    """Aggregated results from Monte Carlo tournament simulations."""
    n_sims: int
    title_probs: dict[str, float] = field(default_factory=dict)
    round_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    final_matchups: dict[tuple[str, str], float] = field(default_factory=dict)


def simulate_tournament(
    bracket: WimbledonBracket,
    predict_fn,
    n_sims: int = MONTE_CARLO_SIMS,
    seed: int = 42,
) -> SimulationResult:
    """Run N tournament simulations using a prediction function.

    Args:
        bracket: The bracket with known results locked in.
        predict_fn: Callable(player_a_id, player_b_id) -> P(A wins).
        n_sims: Number of simulations to run.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with aggregated probabilities.
    """
    rng = np.random.default_rng(seed)

    titles: dict[str, int] = defaultdict(int)
    round_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    final_matchups: dict[tuple[str, str], int] = defaultdict(int)

    for sim in range(n_sims):
        sim_bracket = copy.deepcopy(bracket)

        # Play through each round
        for rnd in ROUNDS:
            playable = [
                m for m in sim_bracket.matches.values()
                if m.round_name == rnd
                and m.player_a is not None
                and m.player_b is not None
                and m.winner is None
            ]

            for match in playable:
                p_a_wins = predict_fn(
                    match.player_a.player_id,
                    match.player_b.player_id,
                )
                if rng.random() < p_a_wins:
                    winner = match.player_a
                else:
                    winner = match.player_b

                match.winner = winner
                sim_bracket._advance_winner(match.match_id, winner)

                # Track round advancement
                round_counts[rnd][winner.name] += 1

            # Track locked results too
            locked = [
                m for m in sim_bracket.matches.values()
                if m.round_name == rnd and m.locked and m.winner
            ]
            for match in locked:
                round_counts[rnd][match.winner.name] += 1

        # Track final matchup
        finals = [m for m in sim_bracket.matches.values() if m.round_name == "F"]
        if finals:
            final = finals[0]
            if final.player_a and final.player_b:
                pair = tuple(sorted([final.player_a.name, final.player_b.name]))
                final_matchups[pair] += 1

        # Track champion
        champ = sim_bracket.get_champion()
        if champ:
            titles[champ.name] += 1

    # Normalize
    result = SimulationResult(n_sims=n_sims)
    result.title_probs = {
        name: count / n_sims
        for name, count in sorted(titles.items(), key=lambda x: -x[1])
    }
    result.round_probs = {
        rnd: {
            name: count / n_sims
            for name, count in sorted(counts.items(), key=lambda x: -x[1])
        }
        for rnd, counts in round_counts.items()
    }
    result.final_matchups = {
        pair: count / n_sims
        for pair, count in sorted(final_matchups.items(), key=lambda x: -x[1])[:20]
    }

    return result
