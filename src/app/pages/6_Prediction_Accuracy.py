"""Prediction accuracy review — daily snapshots, calibration, and title probability history."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR

PREDICTIONS_DIR = DATA_DIR / "predictions"

st.set_page_config(page_title="Prediction Accuracy", layout="wide")
st.title("Prediction accuracy review")
st.caption("Track model performance across daily snapshots")


# ── Load all snapshots ───────────────────────────────────────────────────


@st.cache_data
def load_snapshots() -> dict[str, dict]:
    """Load all prediction snapshot directories."""
    if not PREDICTIONS_DIR.exists():
        return {}
    snapshots = {}
    for d in sorted(PREDICTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        snapshot: dict = {"date": d.name}

        summary_path = d / "accuracy_summary.json"
        if summary_path.exists():
            snapshot["summary"] = json.loads(summary_path.read_text())

        metadata_path = d / "metadata.json"
        if metadata_path.exists():
            snapshot["metadata"] = json.loads(metadata_path.read_text())

        evaluated_path = d / "match_predictions_evaluated.csv"
        if evaluated_path.exists():
            snapshot["evaluated"] = pd.read_csv(evaluated_path)

        title_path = d / "title_probabilities.csv"
        if title_path.exists():
            snapshot["titles"] = pd.read_csv(title_path)

        snapshots[d.name] = snapshot
    return snapshots


snapshots = load_snapshots()

if not snapshots:
    st.warning("No prediction snapshots found in `data/predictions/`.")
    st.stop()


# ── Daily accuracy over time ─────────────────────────────────────────────

st.subheader("Daily accuracy")

accuracy_rows = []
for date, snap in snapshots.items():
    summary = snap.get("summary", {})
    accuracy_rows.append({
        "Date": date,
        "Predictions": summary.get("predictions", 0),
        "Resolved": summary.get("resolved", 0),
        "Correct": summary.get("correct", 0),
        "Accuracy": summary.get("accuracy"),
        "Avg confidence (resolved)": summary.get("average_confidence_resolved"),
    })
accuracy_df = pd.DataFrame(accuracy_rows)

has_resolved = accuracy_df["Resolved"].sum() > 0

if has_resolved:
    resolved_df = accuracy_df[accuracy_df["Accuracy"].notna()].copy()

    col_left, col_right = st.columns(2)
    with col_left:
        fig_acc = px.line(
            resolved_df,
            x="Date",
            y="Accuracy",
            markers=True,
            title="Match prediction accuracy by snapshot date",
        )
        fig_acc.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Baseline (50%)")
        fig_acc.update_layout(
            yaxis_range=[0, 1],
            yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_right:
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(
            x=resolved_df["Date"], y=resolved_df["Correct"],
            name="Correct", marker_color="#00703C",
        ))
        fig_counts.add_trace(go.Bar(
            x=resolved_df["Date"],
            y=resolved_df["Resolved"] - resolved_df["Correct"],
            name="Incorrect", marker_color="#8B0000",
        ))
        fig_counts.update_layout(
            barmode="stack",
            title="Resolved predictions by day",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_counts, use_container_width=True)
else:
    st.info("No predictions have been resolved yet. Results will appear as matches are played.")

st.dataframe(
    accuracy_df.style.format({
        "Accuracy": lambda v: f"{v:.1%}" if pd.notna(v) else "---",
        "Avg confidence (resolved)": lambda v: f"{v:.1%}" if pd.notna(v) else "---",
    }),
    use_container_width=True,
    hide_index=True,
)


# ── Confidence calibration ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Confidence vs actual accuracy")

all_evaluated = pd.concat(
    [snap["evaluated"] for snap in snapshots.values() if "evaluated" in snap],
    ignore_index=True,
)
resolved_matches = all_evaluated[all_evaluated["resolved"]].copy()

if not resolved_matches.empty:
    resolved_matches["confidence_bucket"] = pd.cut(
        resolved_matches["confidence"],
        bins=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
        labels=["50-55%", "55-60%", "60-65%", "65-70%", "70-75%",
                "75-80%", "80-85%", "85-90%", "90-100%"],
    )
    calibration = (
        resolved_matches.groupby("confidence_bucket", observed=True)
        .agg(
            matches=("correct", "count"),
            accuracy=("correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        x=calibration["confidence_bucket"],
        y=calibration["accuracy"],
        name="Actual accuracy",
        marker_color="#00703C",
        text=calibration["matches"].apply(lambda n: f"n={n}"),
        textposition="outside",
    ))
    fig_cal.add_trace(go.Scatter(
        x=calibration["confidence_bucket"],
        y=calibration["avg_confidence"],
        name="Predicted confidence",
        mode="lines+markers",
        line={"color": "#4B2D83", "dash": "dash"},
    ))
    fig_cal.update_layout(
        title="Calibration: predicted confidence vs actual win rate",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.dataframe(
        calibration.style.format({
            "accuracy": "{:.1%}",
            "avg_confidence": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Calibration chart will appear once match predictions are resolved.")


# ── Match-level detail ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("Match predictions detail")

snapshot_dates = sorted(snapshots.keys(), reverse=True)
selected_date = st.selectbox("Snapshot date", snapshot_dates)

snap = snapshots[selected_date]
if "evaluated" in snap:
    detail = snap["evaluated"].copy()

    status_filter = st.radio(
        "Show", ["All", "Resolved only", "Pending only"], horizontal=True,
    )
    if status_filter == "Resolved only":
        detail = detail[detail["resolved"]]
    elif status_filter == "Pending only":
        detail = detail[~detail["resolved"]]

    def color_correct(row):
        if not row["resolved"]:
            return [""] * len(row)
        if row["correct"]:
            return ["background-color: #0a3d1a; color: #4ade80;"] * len(row)
        return ["background-color: #3d0a0a; color: #f87171;"] * len(row)

    display_cols = [
        "match_id", "round", "player_a", "player_b",
        "p_player_a", "p_player_b", "predicted_winner",
        "confidence", "actual_winner", "correct",
    ]
    available_cols = [c for c in display_cols if c in detail.columns]
    styled = detail[available_cols].style.apply(color_correct, axis=1).format({
        "p_player_a": "{:.1%}",
        "p_player_b": "{:.1%}",
        "confidence": "{:.1%}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info(f"No evaluated predictions for {selected_date}.")


# ── Title probability history ────────────────────────────────────────────

st.markdown("---")
st.subheader("Title probability history")

title_rows = []
for date, snap in snapshots.items():
    if "titles" not in snap:
        continue
    for _, row in snap["titles"].iterrows():
        title_rows.append({
            "Date": date,
            "Player": row["player"],
            "P(Title)": row["p_title"],
        })

if title_rows:
    title_history = pd.DataFrame(title_rows)

    # Let user pick players — default to top 8 from most recent snapshot
    latest_date = max(snapshots.keys())
    latest_titles = snapshots[latest_date].get("titles")
    if latest_titles is not None:
        top_players = latest_titles.nlargest(8, "p_title")["player"].tolist()
    else:
        top_players = (
            title_history.groupby("Player")["P(Title)"]
            .max()
            .nlargest(8)
            .index.tolist()
        )

    all_players = sorted(title_history["Player"].unique())
    selected_players = st.multiselect(
        "Players", all_players, default=top_players,
    )

    if selected_players:
        filtered = title_history[title_history["Player"].isin(selected_players)]
        fig_title = px.line(
            filtered,
            x="Date",
            y="P(Title)",
            color="Player",
            markers=True,
            title="Title probability over time",
        )
        fig_title.update_layout(
            yaxis_tickformat=".1%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        )
        st.plotly_chart(fig_title, use_container_width=True)
else:
    st.info("Title probability history will appear after multiple snapshots are saved.")
