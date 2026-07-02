"""Weighted Elo baseline predictor — no training required."""

import numpy as np
import pandas as pd


class WeightedEloPredictor:
    """Predicts match outcomes using WElo delta and logistic curve."""

    def __init__(self, scale: float = 400.0):
        self.scale = scale
        self.name = "Weighted Elo"

    def predict_proba_single(self, welo_a: float, welo_b: float) -> float:
        """P(A wins) given their weighted Elo ratings."""
        return 1.0 / (1.0 + 10 ** ((welo_b - welo_a) / self.scale))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability from feature matrix.

        Expects 'welo_delta' column in X.
        Returns array of shape (n, 2) for sklearn compatibility.
        """
        delta = X["welo_delta"].values
        p_win = 1.0 / (1.0 + 10 ** (-delta / self.scale))
        return np.column_stack([1 - p_win, p_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
