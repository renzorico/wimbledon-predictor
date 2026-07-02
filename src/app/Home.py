"""Wimbledon 2026 Predictor — Streamlit Home Page."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR

st.set_page_config(
    page_title="Wimbledon 2026 Predictor",
    page_icon="🎾",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="color:#00703C; margin-bottom:0;">Wimbledon 2026 Predictor</h1>
        <p style="color:#4B2D83; font-size:1.1rem;">
            Men's Singles — Statistical Match Prediction Model
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Tournament", "Day 3 — R1 in progress")
col2.metric("Models trained", "3 (WElo, LR, XGBoost)")
col3.metric("Matches analyzed", "74,848")


# ── Model comparison summary ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Model performance comparison")

comparison = pd.DataFrame({
    "Model": ["Weighted Elo", "Logistic Regression", "XGBoost"],
    "Accuracy": [0.634, 0.801, 0.897],
    "Log Loss": [0.630, 0.422, 0.242],
    "Brier Score": [0.220, 0.137, 0.074],
    "ROC AUC": [0.695, 0.888, 0.964],
})

col_left, col_right = st.columns(2)

with col_left:
    fig_acc = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        color="Model",
        color_discrete_sequence=["#666", "#4B2D83", "#00703C"],
        title="Accuracy by model",
    )
    fig_acc.update_layout(
        showlegend=False,
        yaxis_range=[0.5, 1.0],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with col_right:
    fig_auc = go.Figure()
    for i, row in comparison.iterrows():
        color = ["#666", "#4B2D83", "#00703C"][i]
        fig_auc.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row["ROC AUC"]],
            name=row["Model"],
            marker_color=color,
        ))
    fig_auc.update_layout(
        title="ROC AUC by model",
        showlegend=False,
        yaxis_range=[0.5, 1.0],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
    )
    st.plotly_chart(fig_auc, use_container_width=True)

st.dataframe(
    comparison.style.highlight_max(
        subset=["Accuracy", "ROC AUC"],
        color="#00703C",
    ).highlight_min(
        subset=["Log Loss", "Brier Score"],
        color="#00703C",
    ),
    use_container_width=True,
    hide_index=True,
)


# ── Known upsets ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("R1 Upsets — Seeds eliminated")

upsets = pd.DataFrame([
    {"Seed": 4, "Player out": "Ben Shelton", "Lost to": "Otto Virtanen (FIN)", "Score": "6-4 3-6 6-7(8) 6-2 7-6(9)"},
    {"Seed": 11, "Player out": "Casper Ruud", "Lost to": "Hubert Hurkacz (POL)", "Score": "6-4 6-2 7-6(7)"},
    {"Seed": 12, "Player out": "Andrey Rublev", "Lost to": "Roman Safiullin (RUS)", "Score": "6-4 6-7(6) 3-6 6-3 7-6(12)"},
    {"Seed": 14, "Player out": "Luciano Darderi", "Lost to": "Ethan Quinn (USA)", "Score": "7-6(7) 7-5 6-2"},
    {"Seed": 26, "Player out": "Cameron Norrie", "Lost to": "Michael Zheng (USA)", "Score": "6-7(7) 6-2 6-7(2) 6-3 7-6(4)"},
    {"Seed": 27, "Player out": "Ugo Humbert", "Lost to": "Zizou Bergs (BEL)", "Score": "6-2 7-5 4-6 3-6 6-3"},
])
st.dataframe(upsets, use_container_width=True, hide_index=True)


# ── Architecture overview ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Project architecture")
st.code(
    """
    Data Pipeline          Feature Engineering       Models              Output
    ┌──────────────┐      ┌──────────────────┐     ┌──────────────┐    ┌────────────────┐
    │ TML-Database │ ──>  │ Elo (surface)    │ ──> │ WElo baseline│──> │ Title probs    │
    │ 74,848 ATP   │      │ Rolling stats    │     │ Logistic Reg │    │ Round probs    │
    │ matches      │      │ H2H records      │     │ XGBoost      │    │ Bracket sim    │
    │ 2000-2026    │      │ Momentum/form    │     │ (tuned)      │    │ Player profiles│
    └──────────────┘      │ 41 delta features│     └──────────────┘    └────────────────┘
                          └──────────────────┘
    """,
    language=None,
)

st.markdown(
    """
    **Key design decisions:**
    - All features are **deltas** (player A - player B) to encode matchup symmetry
    - **TimeSeriesSplit** cross-validation prevents temporal data leakage
    - **Surface-weighted Elo** (40% grass + 35% overall + 25% hard) handles grass scarcity
    - **Monte Carlo simulation** (10K runs) produces probabilistic tournament predictions

    ---
    *Built by [Renzo Rico](https://github.com/renzorico) — July 2026*
    """
)
