"""Wimbledon 2026 Predictor — Streamlit Home Page."""

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, WIMBLEDON_2026_START_DATE
from src.simulation.bracket import MATCHES_PER_ROUND, ROUNDS

PREDICTIONS_DIR = DATA_DIR / "predictions"

WIMBLEDON_GREEN = "#00703C"
WIMBLEDON_PURPLE = "#4B2D83"
PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
)

st.set_page_config(
    page_title="Wimbledon 2026 Predictor",
    page_icon="W",
    layout="wide",
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_latest_snapshot_dir() -> Path | None:
    """Return the most recent snapshot directory, sorted by name (YYYY-MM-DD)."""
    if not PREDICTIONS_DIR.exists():
        return None
    dirs = sorted([d for d in PREDICTIONS_DIR.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


@st.cache_data(ttl=300)
def load_title_probabilities(snapshot_dir: str) -> pd.DataFrame:
    path = Path(snapshot_dir) / "title_probabilities.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_match_predictions(snapshot_dir: str) -> pd.DataFrame:
    path = Path(snapshot_dir) / "match_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_match_predictions_evaluated(snapshot_dir: str) -> pd.DataFrame:
    path = Path(snapshot_dir) / "match_predictions_evaluated.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_metadata(snapshot_dir: str) -> dict:
    path = Path(snapshot_dir) / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def compute_tournament_day() -> int:
    start = datetime.strptime(WIMBLEDON_2026_START_DATE, "%Y-%m-%d").date()
    today = date.today()
    delta = (today - start).days + 1
    return max(1, delta)


# ── Layout ────────────────────────────────────────────────────────────────────

snapshot_dir = get_latest_snapshot_dir()

if snapshot_dir is None:
    st.error("No prediction snapshots found in data/predictions/.")
    st.stop()

snapshot_dir_str = str(snapshot_dir)
title_probs = load_title_probabilities(snapshot_dir_str)
match_preds = load_match_predictions(snapshot_dir_str)
match_eval = load_match_predictions_evaluated(snapshot_dir_str)
metadata = load_metadata(snapshot_dir_str)

# Prefer evaluated if available, fall back to raw predictions
if not match_eval.empty:
    matches_df = match_eval.copy()
else:
    matches_df = match_preds.copy()
    matches_df["actual_winner"] = None
    matches_df["resolved"] = False
    matches_df["correct"] = False

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;">
        <h1 style="color:{WIMBLEDON_GREEN}; margin-bottom:0.25rem; font-size:2.4rem;">
            Wimbledon 2026 Predictor
        </h1>
        <p style="color:{WIMBLEDON_PURPLE}; font-size:1.15rem; margin:0;">
            Men's Singles — AI Match Predictions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tournament_day = compute_tournament_day()
bracket_summary = metadata.get("bracket_summary", {})
locked_total = bracket_summary.get("locked_total", 0)
model_name = metadata.get("model", "XGBoost calibrated")

col1, col2, col3 = st.columns(3)
col1.metric("Tournament day", f"Day {tournament_day}")
col2.metric("Bracket progress", f"{locked_total} / 127 matches locked")
col3.metric("Model", model_name)

st.markdown("---")

# ── Title probabilities ───────────────────────────────────────────────────────

st.subheader("Title probabilities")

if title_probs.empty:
    st.warning("No title probability data found.")
else:
    title_probs_sorted = title_probs.sort_values("p_title", ascending=False).reset_index(drop=True)
    top12 = title_probs_sorted.head(12)

    fig = go.Figure(go.Bar(
        x=top12["p_title"] * 100,
        y=top12["player"],
        orientation="h",
        marker_color=WIMBLEDON_GREEN,
        text=[f"{v:.1f}%" for v in top12["p_title"] * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="P(Title) %",
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=10, r=60, t=20, b=40),
        height=420,
        xaxis=dict(range=[0, min(top12["p_title"].max() * 110, 100)]),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full title probability table", expanded=False):
        display_df = title_probs_sorted[["player", "p_title"]].copy()
        display_df.columns = ["Player", "P(Title)"]
        display_df["P(Title)"] = display_df["P(Title)"].map(lambda v: f"{v * 100:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Match predictions ─────────────────────────────────────────────────────────

st.subheader("Upcoming match predictions")

if matches_df.empty:
    st.warning("No match prediction data found.")
else:
    round_order = {r: i for i, r in enumerate(ROUNDS)}
    matches_df["_round_order"] = matches_df["round"].map(round_order).fillna(99)
    matches_df = matches_df.sort_values(["_round_order", "match_id"]).reset_index(drop=True)

    unique_rounds = matches_df["round"].unique()
    unique_rounds = sorted(unique_rounds, key=lambda r: round_order.get(r, 99))

    for rnd in unique_rounds:
        round_matches = matches_df[matches_df["round"] == rnd].reset_index(drop=True)
        n_expected = MATCHES_PER_ROUND.get(rnd, "?")
        st.markdown(
            f"<h4 style='color:{WIMBLEDON_PURPLE}; margin-bottom:0.5rem;'>"
            f"{rnd} &mdash; {len(round_matches)} prediction(s) of {n_expected}"
            f"</h4>",
            unsafe_allow_html=True,
        )

        for _, row in round_matches.iterrows():
            p_a = float(row["p_player_a"])
            p_b = float(row["p_player_b"])
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            predicted_winner = str(row["predicted_winner"])
            confidence = float(row["confidence"])
            resolved = bool(row.get("resolved", False))
            correct = bool(row.get("correct", False))
            actual_winner = row.get("actual_winner", None)

            # Card background
            if resolved and correct:
                card_bg = "rgba(0,112,60,0.15)"
                border_color = WIMBLEDON_GREEN
                status_text = f"Correct — {actual_winner} won"
                status_color = WIMBLEDON_GREEN
            elif resolved and not correct:
                card_bg = "rgba(180,30,30,0.15)"
                border_color = "#B41E1E"
                status_text = f"Incorrect — {actual_winner} won"
                status_color = "#FF6B6B"
            else:
                card_bg = "rgba(255,255,255,0.04)"
                border_color = "rgba(255,255,255,0.12)"
                status_text = "Pending"
                status_color = "#888888"

            # Stacked bar for probability split
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=[p_a * 100],
                y=[""],
                orientation="h",
                marker_color=WIMBLEDON_GREEN if predicted_winner == player_a else "rgba(75,45,131,0.55)",
                name=player_a,
                hovertemplate=f"{player_a}: {p_a * 100:.1f}%<extra></extra>",
            ))
            bar_fig.add_trace(go.Bar(
                x=[p_b * 100],
                y=[""],
                orientation="h",
                marker_color=WIMBLEDON_GREEN if predicted_winner == player_b else "rgba(75,45,131,0.55)",
                name=player_b,
                hovertemplate=f"{player_b}: {p_b * 100:.1f}%<extra></extra>",
            ))
            bar_fig.update_layout(
                **PLOT_LAYOUT,
                barmode="stack",
                xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=0, r=0, t=0, b=0),
                height=28,
                showlegend=False,
            )

            # Render card
            a_bold = f"<strong>{player_a}</strong>" if predicted_winner == player_a else player_a
            b_bold = f"<strong>{player_b}</strong>" if predicted_winner == player_b else player_b
            a_pct = f"{p_a * 100:.1f}%"
            b_pct = f"{p_b * 100:.1f}%"

            st.markdown(
                f"""
                <div style="
                    background:{card_bg};
                    border:1px solid {border_color};
                    border-radius:8px;
                    padding:0.75rem 1rem 0.4rem;
                    margin-bottom:0.6rem;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="font-size:1rem;">{a_bold} <span style="color:#aaa; font-size:0.85rem;">{a_pct}</span></span>
                        <span style="color:#aaa; font-size:0.8rem; padding:0 0.5rem;">vs</span>
                        <span style="font-size:1rem; text-align:right;">{b_bold} <span style="color:#aaa; font-size:0.85rem;">{b_pct}</span></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div style='margin:-0.8rem 0 0.6rem; padding:0 0.1rem;'>"
                f"<span style='font-size:0.8rem; color:{status_color};'>{status_text}</span>"
                f"<span style='font-size:0.78rem; color:#666; float:right;'>"
                f"Confidence: {confidence * 100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Bracket progress ──────────────────────────────────────────────────────────

st.subheader("Bracket progress")

progress_data = []
for rnd in ROUNDS:
    expected = MATCHES_PER_ROUND[rnd]
    # Count locked matches from metadata or derive from bracket summary
    if rnd == "R1":
        locked = bracket_summary.get("r1_complete", 0)
    else:
        # Count resolved matches in predictions data as a proxy
        resolved_count = 0
        if not matches_df.empty and "resolved" in matches_df.columns:
            resolved_count = int(matches_df[(matches_df["round"] == rnd) & (matches_df["resolved"] == True)].shape[0])
        locked = resolved_count
    progress_data.append({"Round": rnd, "Locked": locked, "Total": expected, "Remaining": expected - locked})

progress_df = pd.DataFrame(progress_data)

cols = st.columns(len(ROUNDS))
for col, (_, row) in zip(cols, progress_df.iterrows()):
    pct = int(row["Locked"] / row["Total"] * 100) if row["Total"] > 0 else 0
    col.metric(
        label=row["Round"],
        value=f"{int(row['Locked'])}/{int(row['Total'])}",
        delta=f"{pct}% complete" if pct > 0 else "Not started",
        delta_color="normal" if pct == 100 else "off",
    )

st.markdown(
    f"""
    ---
    <p style="text-align:center; color:#666; font-size:0.82rem; margin-top:0.5rem;">
        Snapshot: {snapshot_dir.name} &nbsp;|&nbsp;
        Model: {model_name} &nbsp;|&nbsp;
        Built by <a href="https://github.com/renzorico" style="color:{WIMBLEDON_GREEN};">Renzo Rico</a>
    </p>
    """,
    unsafe_allow_html=True,
)
