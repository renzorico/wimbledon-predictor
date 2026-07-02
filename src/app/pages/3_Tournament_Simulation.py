"""Tournament simulation page — full-draw Monte Carlo predictions."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MONTE_CARLO_SIMS, PROCESSED_DIR
from src.simulation.draw_loader import get_bracket_summary, load_wimbledon_2026_draw
from src.simulation.monte_carlo import simulate_tournament
from src.simulation.predictor import CALIBRATED_MODEL_NAME, MatchPredictor

st.set_page_config(page_title="Tournament Simulation", layout="wide")
st.title("Monte Carlo tournament simulation")
st.caption("Full 128-player draw using the trained match model on each unresolved match")


@st.cache_data
def load_matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")


@st.cache_resource
def load_predictor(model_name: str) -> MatchPredictor:
    matches = load_matches()
    return MatchPredictor(
        matches=matches,
        model_name=model_name,
    )


@st.cache_resource
def load_bracket():
    return load_wimbledon_2026_draw()


@st.cache_data(show_spinner=False)
def run_simulation(model_name: str, n_sims: int):
    bracket = load_bracket()
    predictor = load_predictor(model_name)
    return simulate_tournament(
        bracket=bracket,
        predict_fn=predictor.predict_proba,
        n_sims=n_sims,
    )


def build_player_lookup(bracket) -> pd.DataFrame:
    players = pd.DataFrame(
        [
            {
                "Player": player.name,
                "Seed": player.seed,
                "Nation": player.nation,
            }
            for player in bracket.players
        ]
    ).drop_duplicates(subset=["Player"])
    return players


bracket = load_bracket()
summary = get_bracket_summary(bracket)
player_lookup = build_player_lookup(bracket)

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    model_name = st.selectbox(
        "Model",
        [CALIBRATED_MODEL_NAME, "XGBoost", "Logistic Regression", "Weighted Elo"],
        index=0,
    )
with col2:
    n_sims = st.slider("Simulations", 1000, 20000, MONTE_CARLO_SIMS, step=1000)
with col3:
    st.metric("Locked results", summary["locked_total"])

with st.spinner(f"Running {n_sims:,} full-draw simulations with {model_name}..."):
    sim = run_simulation(model_name, n_sims)

title_df = (
    pd.DataFrame(
        [{"Player": name, "P(Title)": prob} for name, prob in sim.title_probs.items()]
    )
    .merge(player_lookup, on="Player", how="left")
    .sort_values("P(Title)", ascending=False)
)

sf_probs = sim.round_probs.get("QF", {})
final_probs = sim.round_probs.get("SF", {})
advancement_df = title_df[["Player", "Seed", "Nation", "P(Title)"]].copy()
advancement_df["P(SF)"] = advancement_df["Player"].map(sf_probs).fillna(0.0)
advancement_df["P(Final)"] = advancement_df["Player"].map(final_probs).fillna(0.0)
advancement_df = advancement_df[
    ["Seed", "Player", "Nation", "P(SF)", "P(Final)", "P(Title)"]
].sort_values("P(Title)", ascending=False)

top_players = title_df.head(16).copy()
chart = px.bar(
    top_players.sort_values("P(Title)", ascending=True),
    x="P(Title)",
    y="Player",
    orientation="h",
    color="P(Title)",
    color_continuous_scale=["#d9ead3", "#38761d"],
    title="Top title probabilities",
)
chart.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=560,
    showlegend=False,
)

left, right = st.columns([1.8, 1])
with left:
    st.plotly_chart(chart, use_container_width=True)
with right:
    st.subheader("Simulation summary")
    st.metric("Draw size", f"{len(player_lookup)} players")
    st.metric("R1 complete", summary["r1_pct"])
    st.metric("Model", model_name)
    st.metric("Reference date", str(load_predictor(model_name).reference_date.date()))
    if model_name == CALIBRATED_MODEL_NAME:
        st.caption("Raw XGBoost probabilities are tempered, blended with WElo, and capped per match.")

st.subheader("Advancement probabilities")
st.dataframe(
    advancement_df.head(24).style.format(
        {
            "P(SF)": "{:.1%}",
            "P(Final)": "{:.1%}",
            "P(Title)": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Most likely finals")
finals_df = pd.DataFrame(
    [
        {"Final": f"{pair[0]} vs {pair[1]}", "Probability": prob}
        for pair, prob in sim.final_matchups.items()
    ]
)
if finals_df.empty:
    st.info("No final matchups available from the current simulation state.")
else:
    st.dataframe(
        finals_df.style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
