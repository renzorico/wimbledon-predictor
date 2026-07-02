"""Model evaluation: metrics, cross-validation, calibration."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.config import CV_SPLITS


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    name: str,
) -> dict:
    """Compute core metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def time_series_cv(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = CV_SPLITS,
) -> dict:
    """Evaluate model using TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        fold_metrics = evaluate_model(model, X_val, y_val, f"fold_{fold}")
        metrics.append(fold_metrics)

    results = pd.DataFrame(metrics)
    return {
        "accuracy_mean": results["accuracy"].mean(),
        "accuracy_std": results["accuracy"].std(),
        "log_loss_mean": results["log_loss"].mean(),
        "log_loss_std": results["log_loss"].std(),
        "brier_mean": results["brier_score"].mean(),
        "brier_std": results["brier_score"].std(),
        "auc_mean": results["roc_auc"].mean(),
        "auc_std": results["roc_auc"].std(),
    }


def get_calibration_data(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction_of_positives, mean_predicted_value) for calibration plot."""
    y_proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, y_proba, n_bins=n_bins)
