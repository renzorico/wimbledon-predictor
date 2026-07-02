"""Refresh data, rebuild models, and save a dated prediction snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean import clean_matches, save_clean_matches
from src.data.download import download_atp_data
from src.data.loader import load_matches
from src.features.builder import build_feature_matrix, save_feature_matrix
from src.models.compare import run_comparison
from src.simulation.predictor import CALIBRATED_MODEL_NAME
from src.simulation.snapshot import write_prediction_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Snapshot date in YYYY-MM-DD format.")
    parser.add_argument("--model", default=CALIBRATED_MODEL_NAME)
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Refresh data and write predictions without retraining models.",
    )
    args = parser.parse_args()

    print("refreshing live TennisMyLife data...")
    download_atp_data()

    print("cleaning matches...")
    matches = clean_matches(load_matches())
    save_clean_matches(matches)

    print("building feature matrix...")
    features = build_feature_matrix(matches)
    save_feature_matrix(features)

    if not args.skip_train:
        print("retraining models...")
        run_comparison(features)

    print("writing prediction snapshot...")
    out_dir = write_prediction_snapshot(
        snapshot_date=args.date,
        model_name=args.model,
        n_sims=args.sims,
    )
    print(f"saved prediction snapshot to {out_dir}")


if __name__ == "__main__":
    main()
