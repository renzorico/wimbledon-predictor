"""Model comparison page — metrics, feature importance, calibration."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.models.elo_model import WeightedEloPredictor
from src.models.evaluate import evaluate_model, get_calibration_data
from src.models.compare import get_feature_columns, _flip_rows
from src.config import RANDOM_STATE, TEST_YEAR_CUTOFF

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model comparison")
st.caption("WElo baseline vs Logistic Regression vs XGBoost (tuned)")

# ── Load data & models ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    feature_cols = get_feature_columns(features)
    test = features[features["tourney_date"].dt.year >= TEST_YEAR_CUTOFF]
    test = _flip_rows(test, feature_cols, RANDOM_STATE + 1)
    X_test = test[feature_cols].fillna(0)
    y_test = test["y"].values
    return X_test, y_test, feature_cols

@st.cache_resource
def load_models():
    elo = WeightedEloPredictor()
    lr = joblib.load(MODELS_DIR / "logistic_pipeline.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_pipeline.pkl")
    return {"Weighted Elo": elo, "Logistic Regression": lr, "XGBoost": xgb}

X_test, y_test, feature_cols = load_data()
models = load_models()

# ── Metrics table ─────────────────────────────────────────────────────────
st.subheader("Test set metrics")
st.caption(f"Test set: {len(X_test):,} matches ({TEST_YEAR_CUTOFF}+)")

metrics = []
for name, model in models.items():
    m = evaluate_model(model, X_test, y_test, name)
    metrics.append(m)

metrics_df = pd.DataFrame(metrics)
st.dataframe(
    metrics_df.style
        .highlight_max(subset=["accuracy", "roc_auc"], color="#00703C")
        .highlight_min(subset=["log_loss", "brier_score"], color="#00703C")
        .format({
            "accuracy": "{:.1%}",
            "log_loss": "{:.4f}",
            "brier_score": "{:.4f}",
            "roc_auc": "{:.4f}",
        }),
    use_container_width=True,
    hide_index=True,
)


# ── Calibration curves ───────────────────────────────────────────────────
st.subheader("Calibration curves")
st.caption("A well-calibrated model's curve should hug the diagonal.")

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(dash="dash", color="gray"),
    name="Perfect calibration",
))

colors = {"Weighted Elo": "#666", "Logistic Regression": "#4B2D83", "XGBoost": "#00703C"}
for name, model in models.items():
    frac_pos, mean_pred = get_calibration_data(model, X_test, y_test, n_bins=10)
    fig_cal.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos, mode="lines+markers",
        name=name, line=dict(color=colors[name]),
    ))

fig_cal.update_layout(
    xaxis_title="Mean predicted probability",
    yaxis_title="Fraction of positives",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
    height=450,
)
st.plotly_chart(fig_cal, use_container_width=True)


# ── Feature importance (XGBoost) ─────────────────────────────────────────
st.subheader("Feature importance (XGBoost)")

xgb_model = models["XGBoost"]
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=True).tail(15)

fig_imp = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h",
    color_discrete_sequence=["#00703C"],
    title="Top 15 features by gain",
)
fig_imp.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
    height=500,
    yaxis_title="",
)
st.plotly_chart(fig_imp, use_container_width=True)


# ── Logistic Regression coefficients ──────────────────────────────────────
st.subheader("Feature coefficients (Logistic Regression)")

lr_model = models["Logistic Regression"]
coefs = lr_model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coefs,
}).sort_values("coefficient", ascending=True)

# Show top and bottom
top_bottom = pd.concat([coef_df.head(8), coef_df.tail(8)])

fig_coef = px.bar(
    top_bottom,
    x="coefficient",
    y="feature",
    orientation="h",
    color="coefficient",
    color_continuous_scale=["#ff4444", "#666", "#00703C"],
    title="Strongest positive and negative coefficients",
)
fig_coef.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
    height=500,
    yaxis_title="",
)
st.plotly_chart(fig_coef, use_container_width=True)
