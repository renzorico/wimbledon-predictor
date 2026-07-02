"""Tournament simulation page — Monte Carlo bracket predictions."""

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
from src.features.elo import compute_elo_ratings
from src.models.elo_model import WeightedEloPredictor

st.set_page_config(page_title="Tournament Simulation", layout="wide")
st.title("Monte Carlo tournament simulation")
st.caption("Simulating the Wimbledon 2026 bracket using trained models")


# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_elo_ratings():
    matches = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    return compute_elo_ratings(matches)


@st.cache_resource
def load_xgb():
    return joblib.load(MODELS_DIR / "xgb_pipeline.pkl")


# ── Seeds data ────────────────────────────────────────────────────────────
SEEDS_DATA = [
    (1, "Jannik Sinner", "ITA", "Q1"),
    (2, "Alexander Zverev", "GER", "Q4"),
    (3, "Felix Auger-Aliassime", "CAN", "Q2"),
    (4, "Ben Shelton", "USA", "Q3"),
    (5, "Alex de Minaur", "AUS", "Q1"),
    (6, "Taylor Fritz", "USA", "Q4"),
    (7, "Novak Djokovic", "SRB", "Q2"),
    (8, "Daniil Medvedev", "RUS", "Q3"),
    (9, "Flavio Cobolli", "ITA", "Q1"),
    (10, "Alexander Bublik", "KAZ", "Q4"),
    (11, "Casper Ruud", "NOR", "Q2"),
    (12, "Andrey Rublev", "RUS", "Q3"),
    (13, "Jiri Lehecka", "CZE", "Q1"),
    (14, "Luciano Darderi", "ITA", "Q4"),
    (15, "Jakub Mensik", "CZE", "Q2"),
    (16, "Learner Tien", "USA", "Q3"),
    (17, "Frances Tiafoe", "USA", "Q1"),
    (18, "Francisco Cerundolo", "ARG", "Q4"),
    (19, "Karen Khachanov", "RUS", "Q2"),
    (20, "Arthur Fils", "FRA", "Q3"),
    (21, "Tommy Paul", "USA", "Q1"),
    (22, "Alejandro Davidovich Fokina", "ESP", "Q4"),
    (23, "Rafael Jodar", "ESP", "Q2"),
    (24, "Joao Fonseca", "BRA", "Q3"),
    (25, "Arthur Rinderknech", "FRA", "Q1"),
    (26, "Cameron Norrie", "GBR", "Q4"),
    (27, "Ugo Humbert", "FRA", "Q2"),
    (28, "Brandon Nakashima", "USA", "Q3"),
    (29, "Tomas Martin Etcheverry", "ARG", "Q1"),
    (30, "Alejandro Tabilo", "CHI", "Q4"),
    (31, "Ignacio Buse", "PER", "Q2"),
    (32, "Matteo Arnaldi", "ITA", "Q3"),
]

ELIMINATED = {4, 11, 12, 14, 26, 27}  # Seeds eliminated in R1

# ── Simulation engine ─────────────────────────────────────────────────────

def run_elo_simulation(
    elo_ratings: dict,
    n_sims: int,
    seeds_data: list,
    eliminated: set,
) -> dict[str, float]:
    """Simple Monte Carlo using Elo-based probabilities among active seeds."""
    rng = np.random.default_rng(42)
    elo_pred = WeightedEloPredictor()

    # Map names to IDs in elo_ratings
    name_to_id: dict[str, str] = {}
    matches = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    for _, row in matches.iterrows():
        name_to_id[row["winner_name"]] = str(row["winner_id"])
        name_to_id[row["loser_name"]] = str(row["loser_id"])

    active_seeds = [
        (s, n, nat, q)
        for s, n, nat, q in seeds_data
        if s not in eliminated
    ]

    # Get Elo for each active seed
    seed_elos: dict[str, float] = {}
    for seed, name, nat, quarter in active_seeds:
        pid = name_to_id.get(name)
        if pid and pid in elo_ratings:
            seed_elos[name] = elo_ratings[pid].welo
        else:
            seed_elos[name] = 1500  # default

    # Simulate quarter → semi → final for seeds only
    title_counts: dict[str, int] = {}
    for name in seed_elos:
        title_counts[name] = 0

    quarters = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for s, n, nat, q in active_seeds:
        quarters[q].append(n)

    for _ in range(n_sims):
        # Quarter finals: pick winner per quarter
        qf_winners = {}
        for q, players in quarters.items():
            if not players:
                continue
            # Simple: probability proportional to Elo
            elos = np.array([seed_elos[p] for p in players])
            probs = np.exp((elos - elos.max()) / 200)
            probs = probs / probs.sum()
            winner = rng.choice(players, p=probs)
            qf_winners[q] = winner

        # Semis
        sf1_players = [qf_winners.get("Q1"), qf_winners.get("Q2")]
        sf2_players = [qf_winners.get("Q3"), qf_winners.get("Q4")]
        sf1_players = [p for p in sf1_players if p]
        sf2_players = [p for p in sf2_players if p]

        def pick_winner(players):
            if len(players) == 1:
                return players[0]
            e = np.array([seed_elos[p] for p in players])
            p_win = 1 / (1 + 10 ** ((e[1] - e[0]) / 400))
            return players[0] if rng.random() < p_win else players[1]

        sf1_w = pick_winner(sf1_players) if sf1_players else None
        sf2_w = pick_winner(sf2_players) if sf2_players else None

        # Final
        finalists = [p for p in [sf1_w, sf2_w] if p]
        if finalists:
            champion = pick_winner(finalists) if len(finalists) == 2 else finalists[0]
            title_counts[champion] = title_counts.get(champion, 0) + 1

    return {k: v / n_sims for k, v in sorted(title_counts.items(), key=lambda x: -x[1])}


# ── UI controls ───────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    n_sims = st.slider("Simulations", 1000, 50000, 10000, step=1000)

with st.spinner("Computing Elo ratings..."):
    elo_ratings = load_elo_ratings()

with st.spinner(f"Running {n_sims:,} simulations..."):
    title_probs = run_elo_simulation(elo_ratings, n_sims, SEEDS_DATA, ELIMINATED)

# ── Results ───────────────────────────────────────────────────────────────
st.subheader("Title probabilities")

prob_df = pd.DataFrame([
    {"Seed": next((s for s, n, _, _ in SEEDS_DATA if n == name), "?"),
     "Player": name,
     "P(Title)": prob}
    for name, prob in title_probs.items()
    if prob > 0.001
]).sort_values("P(Title)", ascending=False).head(20)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig = px.bar(
        prob_df.head(12),
        x="P(Title)",
        y="Player",
        orientation="h",
        color="P(Title)",
        color_continuous_scale=["#4B2D83", "#00703C"],
        title="Top 12 title contenders",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        yaxis=dict(autorange="reversed"),
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.dataframe(
        prob_df.style.format({"P(Title)": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

# ── Quarter analysis ─────────────────────────────────────────────────────
st.subheader("Quarter-by-quarter analysis")

for q_label, q_name in [("Q1", "Sinner half"), ("Q2", "Auger-Aliassime half"),
                          ("Q3", "Medvedev half"), ("Q4", "Zverev half")]:
    q_seeds = [
        (s, n, nat) for s, n, nat, q in SEEDS_DATA
        if q == q_label and s not in ELIMINATED
    ]
    with st.expander(f"{q_label} — {q_name} ({len(q_seeds)} active seeds)"):
        q_data = []
        for s, n, nat in q_seeds:
            welo = elo_ratings.get(
                next((name_to_id for name_to_id in [n]), None),
                None
            )
            q_data.append({"Seed": s, "Player": n, "Nation": nat})
        st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)
