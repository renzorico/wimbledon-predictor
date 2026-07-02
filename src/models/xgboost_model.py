"""XGBoost pipeline with hyperparameter tuning."""

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import (
    CV_SPLITS,
    RANDOM_STATE,
    XGB_SEARCH_ITERS,
    XGB_SEARCH_SPACE,
)


def build_xgb_base() -> XGBClassifier:
    """Create a base XGBClassifier with sensible defaults."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def build_xgb_tuned(X_train, y_train) -> XGBClassifier:
    """Tune XGBClassifier with RandomizedSearchCV + TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    base = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=XGB_SEARCH_ITERS,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"best params: {search.best_params_}")
    print(f"best log loss: {-search.best_score_:.4f}")

    return search.best_estimator_
