"""Wimbledon 2026 Predictor — Home Page."""

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_analytics2 as streamlit_analytics

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, WIMBLEDON_2026_START_DATE
from src.simulation.bracket import MATCHES_PER_ROUND, ROUNDS

ANALYTICS_FILE = DATA_DIR / "analytics.json"

PREDICTIONS_DIR = DATA_DIR / "predictions"

GREEN = "#00703C"
PURPLE = "#4B2D83"
CARD_BG = "#F9F9F9"
CARD_BORDER = "#E0E0E0"
TEXT_DARK = "#1A1A1A"
TEXT_MUTED = "#666666"

PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color=TEXT_DARK,
)

st.set_page_config(
    page_title="Wimbledon 2026 Predictor",
    page_icon="W",
    layout="wide",
)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_latest_snapshot_dir() -> Path | None:
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


# ── Analytics ─────────────────────────────────────────────────────────────────

streamlit_analytics.start_tracking(save_to_json=str(ANALYTICS_FILE))

# ── Load data ────────────────────────────────────────────────────────────────

snapshot_dir = get_latest_snapshot_dir()

if snapshot_dir is None:
    st.error("No prediction snapshots found in data/predictions/.")
    st.stop()

snapshot_dir_str = str(snapshot_dir)
title_probs = load_title_probabilities(snapshot_dir_str)
match_preds = load_match_predictions(snapshot_dir_str)
match_eval = load_match_predictions_evaluated(snapshot_dir_str)
metadata = load_metadata(snapshot_dir_str)

if not match_eval.empty:
    matches_df = match_eval.copy()
else:
    matches_df = match_preds.copy()
    matches_df["actual_winner"] = None
    matches_df["resolved"] = False
    matches_df["correct"] = False

tournament_day = compute_tournament_day()
bracket_summary = metadata.get("bracket_summary", {})
locked_total = bracket_summary.get("locked_total", 0)
model_name = metadata.get("model", "XGBoost calibrated")


# ── 1. Hero banner ────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {GREEN} 0%, #004d29 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="
            font-size: 0.85rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.7);
            margin-bottom: 0.6rem;
            font-weight: 500;
        ">Men's Singles</div>
        <h1 style="
            color: #FFFFFF;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0 0 0.5rem;
            letter-spacing: -0.5px;
            line-height: 1.1;
        ">Wimbledon 2026 Predictor</h1>
        <p style="
            color: rgba(255,255,255,0.85);
            font-size: 1.1rem;
            margin: 0 0 1.8rem;
            font-weight: 400;
        ">AI-powered match predictions &middot; Updated daily</p>
        <div style="
            display: inline-flex;
            gap: 0;
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            overflow: hidden;
        ">
            <div style="padding: 0.65rem 1.4rem; border-right: 1px solid rgba(255,255,255,0.15);">
                <div style="color: rgba(255,255,255,0.6); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">Tournament Day</div>
                <div style="color: #FFFFFF; font-size: 1.05rem; font-weight: 600;">Day {tournament_day}</div>
            </div>
            <div style="padding: 0.65rem 1.4rem; border-right: 1px solid rgba(255,255,255,0.15);">
                <div style="color: rgba(255,255,255,0.6); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">Matches Completed</div>
                <div style="color: #FFFFFF; font-size: 1.05rem; font-weight: 600;">{locked_total} of 127</div>
            </div>
            <div style="padding: 0.65rem 1.4rem;">
                <div style="color: rgba(255,255,255,0.6); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">Model</div>
                <div style="color: #FFFFFF; font-size: 1.05rem; font-weight: 600;">{model_name}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── 2. Top 3 contenders ───────────────────────────────────────────────────────

if not title_probs.empty:
    title_probs_sorted = title_probs.sort_values("p_title", ascending=False).reset_index(drop=True)
    top3 = title_probs_sorted.head(3)

    medal_labels = ["Favourite", "2nd Favourite", "3rd Favourite"]
    medal_colors = [GREEN, PURPLE, "#B8860B"]
    medal_font_sizes = ["2.6rem", "2.2rem", "2rem"]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        pct = row["p_title"] * 100
        border_color = medal_colors[i]
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #FFFFFF;
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    padding: 1.5rem 1.2rem 1.4rem;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    margin-bottom: 1.5rem;
                ">
                    <div style="
                        font-size: 0.72rem;
                        text-transform: uppercase;
                        letter-spacing: 0.14em;
                        color: {border_color};
                        font-weight: 700;
                        margin-bottom: 0.6rem;
                    ">{medal_labels[i]}</div>
                    <div style="
                        font-size: 1.2rem;
                        font-weight: 700;
                        color: {TEXT_DARK};
                        margin-bottom: 0.7rem;
                        line-height: 1.2;
                    ">{row["player"]}</div>
                    <div style="
                        font-size: {medal_font_sizes[i]};
                        font-weight: 800;
                        color: {border_color};
                        line-height: 1;
                    ">{pct:.1f}%</div>
                    <div style="
                        font-size: 0.75rem;
                        color: {TEXT_MUTED};
                        margin-top: 0.4rem;
                    ">chance of winning title</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── 3. Full title probability chart ──────────────────────────────────────────

st.markdown(
    f"<h3 style='color: {TEXT_DARK}; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.2rem;'>"
    "Title Probabilities &mdash; Top 12"
    "</h3>",
    unsafe_allow_html=True,
)

if title_probs.empty:
    st.warning("No title probability data found.")
else:
    top12 = title_probs_sorted.head(12).iloc[::-1].reset_index(drop=True)

    bar_colors = [GREEN if i >= len(top12) - 3 else "#A8D5BA" for i in range(len(top12))]

    fig = go.Figure(go.Bar(
        x=top12["p_title"] * 100,
        y=top12["player"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
        ),
        text=[f"  {v:.1f}%" for v in top12["p_title"] * 100],
        textposition="outside",
        textfont=dict(size=12, color=TEXT_DARK),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(
            range=[0, min(top12["p_title"].max() * 120, 100)],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=13, color=TEXT_DARK),
            showgrid=False,
        ),
        margin=dict(l=10, r=80, t=10, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full table", expanded=False):
        display_df = title_probs_sorted[["player", "p_title"]].copy()
        display_df.columns = ["Player", "P(Title)"]
        display_df["P(Title)"] = display_df["P(Title)"].map(lambda v: f"{v * 100:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)


# ── 4. Match predictions ──────────────────────────────────────────────────────

st.markdown(
    f"<h3 style='color: {TEXT_DARK}; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem;'>"
    "Match Predictions"
    "</h3>",
    unsafe_allow_html=True,
)

if matches_df.empty:
    st.warning("No match prediction data found.")
else:
    round_order = {r: i for i, r in enumerate(ROUNDS)}
    matches_df["_round_order"] = matches_df["round"].map(round_order).fillna(99)
    matches_df = matches_df.sort_values(["_round_order", "match_id"]).reset_index(drop=True)

    unique_rounds = sorted(matches_df["round"].unique(), key=lambda r: round_order.get(r, 99))

    ROUND_LABELS = {
        "R1": "Round 1", "R2": "Round 2", "R3": "Round 3", "R4": "Round 4",
        "QF": "Quarter-Finals", "SF": "Semi-Finals", "F": "Final",
    }

    for rnd in unique_rounds:
        round_matches = matches_df[matches_df["round"] == rnd].reset_index(drop=True)
        n_expected = MATCHES_PER_ROUND.get(rnd, "?")
        label = ROUND_LABELS.get(rnd, rnd)

        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 1.5rem 0 0.75rem;
            ">
                <span style="
                    background: {PURPLE};
                    color: #FFFFFF;
                    font-size: 0.7rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    padding: 0.25rem 0.65rem;
                    border-radius: 4px;
                ">{rnd}</span>
                <span style="
                    font-size: 1rem;
                    font-weight: 700;
                    color: {TEXT_DARK};
                ">{label}</span>
                <span style="
                    font-size: 0.8rem;
                    color: {TEXT_MUTED};
                ">{len(round_matches)} of {n_expected} predictions</span>
            </div>
            """,
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

            a_is_winner = predicted_winner == player_a

            if resolved and correct:
                card_border = GREEN
                card_accent_bg = "rgba(0,112,60,0.06)"
                status_html = (
                    f"<span style='color:{GREEN}; font-weight:600;'>Correct</span>"
                    f"<span style='color:{TEXT_MUTED};'> &mdash; {actual_winner} won</span>"
                )
            elif resolved and not correct:
                card_border = "#C0392B"
                card_accent_bg = "rgba(192,57,43,0.05)"
                status_html = (
                    f"<span style='color:#C0392B; font-weight:600;'>Incorrect</span>"
                    f"<span style='color:{TEXT_MUTED};'> &mdash; {actual_winner} won</span>"
                )
            else:
                card_border = CARD_BORDER
                card_accent_bg = CARD_BG
                status_html = (
                    f"<span style='color:{TEXT_MUTED};'>Prediction pending</span>"
                )

            a_name_style = f"font-weight:700; color:{GREEN};" if a_is_winner else f"color:{TEXT_DARK};"
            b_name_style = f"font-weight:700; color:{GREEN};" if not a_is_winner else f"color:{TEXT_DARK};"
            a_pct_style = f"color:{GREEN}; font-weight:700;" if a_is_winner else f"color:{TEXT_MUTED};"
            b_pct_style = f"color:{GREEN}; font-weight:700;" if not a_is_winner else f"color:{TEXT_MUTED};"

            bar_a_pct = round(p_a * 100, 1)
            bar_b_pct = round(p_b * 100, 1)

            st.markdown(
                f"""
                <div style="
                    background: {card_accent_bg};
                    border: 1px solid {card_border};
                    border-radius: 10px;
                    padding: 1rem 1.1rem 0.85rem;
                    margin-bottom: 0.6rem;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.65rem;
                    ">
                        <div style="flex: 1; min-width: 0;">
                            <div style="font-size: 0.98rem; {a_name_style} white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{player_a}</div>
                            <div style="font-size: 0.88rem; {a_pct_style} margin-top: 0.1rem;">{bar_a_pct:.1f}%</div>
                        </div>
                        <div style="
                            flex-shrink: 0;
                            padding: 0 0.9rem;
                            font-size: 0.78rem;
                            font-weight: 600;
                            color: {TEXT_MUTED};
                            text-transform: uppercase;
                            letter-spacing: 0.08em;
                        ">vs</div>
                        <div style="flex: 1; min-width: 0; text-align: right;">
                            <div style="font-size: 0.98rem; {b_name_style} white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{player_b}</div>
                            <div style="font-size: 0.88rem; {b_pct_style} margin-top: 0.1rem;">{bar_b_pct:.1f}%</div>
                        </div>
                    </div>
                    <div style="
                        height: 6px;
                        border-radius: 3px;
                        overflow: hidden;
                        background: {PURPLE};
                        margin-bottom: 0.65rem;
                    ">
                        <div style="
                            width: {bar_a_pct}%;
                            height: 100%;
                            background: {GREEN};
                            border-radius: 3px 0 0 3px;
                        "></div>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div style="font-size: 0.78rem;">{status_html}</div>
                        <div style="font-size: 0.75rem; color: {TEXT_MUTED};">
                            Confidence: {confidence * 100:.1f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid {CARD_BORDER};
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 0.8rem;
    ">
        Snapshot: {snapshot_dir.name}
        &nbsp;&middot;&nbsp;
        Model: {model_name}
        &nbsp;&middot;&nbsp;
        Built by <a href="https://github.com/renzorico" style="color:{GREEN}; text-decoration:none;">Renzo Rico</a>
    </div>
    """,
    unsafe_allow_html=True,
)

streamlit_analytics.stop_tracking(save_to_json=str(ANALYTICS_FILE))
