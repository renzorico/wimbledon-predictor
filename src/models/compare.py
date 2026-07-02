"""Side-by-side model comparison."""

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RANDOM_STATE, TEST_YEAR_CUTOFF
from src.models.elo_model import WeightedEloPredictor
from src.models.evaluate import evaluate_model
from src.models.logistic import build_logistic_pipeline
from src.models.xgboost_model import build_xgb_tuned


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns from the feature matrix."""
    delta_cols = [c for c in df.columns if c.endswith("_delta")]
    h2h_cols = [c for c in df.columns if c.startswith("h2h_")]
    extra = ["surface_grass"]
    return [c for c in delta_cols + h2h_cols + extra if c in df.columns]


def _flip_rows(df: pd.DataFrame, feature_cols: list[str], seed: int) -> pd.DataFrame:
    """Randomly flip half the rows (swap winner/loser perspective).

    This teaches the model symmetry: it shouldn't always predict 'winner wins'
    since we always construct features from winner perspective.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < 0.5

    flipped = df.copy()
    for col in feature_cols:
        if col.startswith("h2h_") or col == "surface_grass":
            continue
        flipped.loc[mask, col] = -flipped.loc[mask, col]

    # Flip H2H columns
    if "h2h_win_pct_w" in flipped.columns:
        flipped.loc[mask, "h2h_win_pct_w"] = 1 - flipped.loc[mask, "h2h_win_pct_w"]
    if "h2h_grass_win_pct_w" in flipped.columns:
        flipped.loc[mask, "h2h_grass_win_pct_w"] = 1 - flipped.loc[mask, "h2h_grass_win_pct_w"]
    if "h2h_recent_3_w" in flipped.columns:
        flipped.loc[mask, "h2h_recent_3_w"] = 1 - flipped.loc[mask, "h2h_recent_3_w"]

    flipped.loc[mask, "y"] = 0

    return flipped


def run_comparison(features: pd.DataFrame) -> pd.DataFrame:
    """Train all models, evaluate, return comparison table."""
    feature_cols = get_feature_columns(features)
    print(f"using {len(feature_cols)} features: {feature_cols[:5]}...")

    # Split by year
    train = features[features["tourney_date"].dt.year < TEST_YEAR_CUTOFF]
    test = features[features["tourney_date"].dt.year >= TEST_YEAR_CUTOFF]
    print(f"train: {len(train):,} rows | test: {len(test):,} rows")

    # Flip training rows for symmetry
    train = _flip_rows(train, feature_cols, RANDOM_STATE)

    X_train = train[feature_cols].fillna(0)
    y_train = train["y"].values
    X_test = test[feature_cols].fillna(0)
    y_test = test["y"].values

    # Also flip test set for fair evaluation
    test_flipped = _flip_rows(test, feature_cols, RANDOM_STATE + 1)
    X_test = test_flipped[feature_cols].fillna(0)
    y_test = test_flipped["y"].values

    results: list[dict] = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Weighted Elo baseline
    print("\n[1/3] Weighted Elo baseline...")
    elo_model = WeightedEloPredictor()
    elo_metrics = evaluate_model(elo_model, X_test, y_test, "Weighted Elo")
    results.append(elo_metrics)
    print(f"  accuracy: {elo_metrics['accuracy']:.4f}")

    # 2. Logistic Regression
    print("\n[2/3] Logistic Regression...")
    lr_pipeline = build_logistic_pipeline()
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
    results.append(lr_metrics)
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_pipeline.pkl")
    print(f"  accuracy: {lr_metrics['accuracy']:.4f}")

    # 3. XGBoost (tuned)
    print("\n[3/3] XGBoost (tuning)...")
    xgb_model = build_xgb_tuned(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_metrics)
    joblib.dump(xgb_model, MODELS_DIR / "xgb_pipeline.pkl")
    print(f"  accuracy: {xgb_metrics['accuracy']:.4f}")

    # Save feature column order
    import json
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    comparison = pd.DataFrame(results)
    print("\n" + comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    from src.config import PROCESSED_DIR

    features = pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    run_comparison(features)
