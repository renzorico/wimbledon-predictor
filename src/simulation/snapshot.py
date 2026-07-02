"""Persist dated prediction snapshots for later accuracy review."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, MONTE_CARLO_SIMS, PROCESSED_DIR
from src.simulation.draw_loader import get_bracket_summary, load_wimbledon_2026_draw
from src.simulation.monte_carlo import simulate_tournament
from src.simulation.predictor import CALIBRATED_MODEL_NAME, MatchPredictor

PREDICTIONS_DIR = DATA_DIR / "predictions"


def write_prediction_snapshot(
    snapshot_date: str | None = None,
    model_name: str = CALIBRATED_MODEL_NAME,
    n_sims: int = MONTE_CARLO_SIMS,
) -> Path:
    """Write current bracket and tournament predictions to a dated folder."""
    if snapshot_date is None:
        snapshot_date = datetime.now().strftime("%Y-%m-%d")

    matches = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    bracket = load_wimbledon_2026_draw()
    predictor = MatchPredictor(
        matches=matches,
        model_name=model_name,
    )
    result = simulate_tournament(
        bracket=bracket,
        predict_fn=predictor.predict_proba,
        n_sims=n_sims,
    )

    out_dir = PREDICTIONS_DIR / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    match_rows = []
    for match in bracket.get_playable_matches():
        if match.player_a is None or match.player_b is None:
            continue
        p_a = predictor.predict_proba(match.player_a.player_id, match.player_b.player_id)
        match_rows.append(
            {
                "snapshot_date": snapshot_date,
                "match_id": match.match_id,
                "round": match.round_name,
                "player_a": match.player_a.name,
                "player_b": match.player_b.name,
                "p_player_a": p_a,
                "p_player_b": 1 - p_a,
                "predicted_winner": match.player_a.name if p_a >= 0.5 else match.player_b.name,
                "confidence": max(p_a, 1 - p_a),
                "model": model_name,
            }
        )
    pd.DataFrame(match_rows).to_csv(out_dir / "match_predictions.csv", index=False)

    pd.DataFrame(
        [
            {"snapshot_date": snapshot_date, "player": player, "p_title": probability}
            for player, probability in result.title_probs.items()
        ]
    ).to_csv(out_dir / "title_probabilities.csv", index=False)

    pd.DataFrame(
        [
            {
                "snapshot_date": snapshot_date,
                "finalist_a": pair[0],
                "finalist_b": pair[1],
                "probability": probability,
            }
            for pair, probability in result.final_matchups.items()
        ]
    ).to_csv(out_dir / "final_matchups.csv", index=False)

    metadata = {
        "snapshot_date": snapshot_date,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "n_sims": n_sims,
        "reference_date": str(predictor.reference_date.date()),
        "history_max_date": str(matches["tourney_date"].max().date()),
        "bracket_summary": get_bracket_summary(bracket),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir
