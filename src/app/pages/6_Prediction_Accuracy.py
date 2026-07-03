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
            font_color="#1A1A1A",
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
            font_color="#1A1A1A",
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
        font_color="#1A1A1A",
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

    ROUND_ORDER = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "QF": 5, "SF": 6, "F": 7}
    detail = detail.copy()
    detail["_round_order"] = detail["round"].map(ROUND_ORDER).fillna(9)
    detail = detail.sort_values(["_round_order", "match_id"]).reset_index(drop=True)

    for rnd, group in detail.groupby("round", sort=False):
        st.markdown(
            f"<p style='font-weight:600; color:#4B2D83; margin:1rem 0 0.4rem;'>{rnd}</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, (_, row) in enumerate(group.iterrows()):
            p_a = float(row["p_player_a"])
            p_b = float(row["p_player_b"])
            pred_winner = str(row["predicted_winner"])
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            resolved = bool(row.get("resolved", False))
            correct = bool(row.get("correct", False))
            actual = row.get("actual_winner", None)

            if resolved and correct:
                border = "#00703C"
                status_html = f"<span style='color:#00703C; font-weight:600;'>Correct</span> — {actual} won"
            elif resolved and not correct:
                border = "#C62828"
                status_html = f"<span style='color:#C62828; font-weight:600;'>Incorrect</span> — {actual} won"
            else:
                border = "#E0E0E0"
                status_html = "<span style='color:#9E9E9E;'>Pending</span>"

            a_bold = f"<strong>{player_a}</strong>" if pred_winner == player_a else player_a
            b_bold = f"<strong>{player_b}</strong>" if pred_winner == player_b else player_b
            a_color = "#00703C" if pred_winner == player_a else "#555"
            b_color = "#00703C" if pred_winner == player_b else "#555"

            card = f"""
            <div style="border:1.5px solid {border}; border-radius:8px; padding:10px 14px;
                        margin-bottom:8px; background:#FAFAFA; font-family:sans-serif;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="color:{a_color}; font-size:14px;">{a_bold}</span>
                <span style="font-size:12px; color:#999; font-weight:600;">{p_a:.0%}</span>
              </div>
              <div style="width:100%; height:6px; background:#E8E8E8; border-radius:3px; margin-bottom:6px;">
                <div style="width:{p_a*100:.1f}%; height:100%; background:#00703C; border-radius:3px;"></div>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="color:{b_color}; font-size:14px;">{b_bold}</span>
                <span style="font-size:12px; color:#999; font-weight:600;">{p_b:.0%}</span>
              </div>
              <div style="font-size:11px; color:#888; margin-top:2px;">{status_html}
                <span style="float:right;">Confidence: {max(p_a, p_b):.0%}</span>
              </div>
            </div>
            """
            with cols[i % 2]:
                st.markdown(card, unsafe_allow_html=True)
else:
    st.info(f"No evaluated predictions for {selected_date}.")


# ── Title odds leaderboard ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Title odds leaderboard")

latest_date = max(snapshots.keys())
latest_titles = snapshots[latest_date].get("titles")

if latest_titles is not None:
    # Build leaderboard with delta vs previous snapshot if available
    sorted_dates = sorted(snapshots.keys())
    prev_titles = None
    if len(sorted_dates) >= 2:
        prev_date = sorted_dates[-2]
        prev_titles = snapshots[prev_date].get("titles")

    latest_sorted = latest_titles.sort_values("p_title", ascending=False).reset_index(drop=True)

    prev_lookup = {}
    if prev_titles is not None:
        prev_lookup = dict(zip(prev_titles["player"], prev_titles["p_title"]))

    rows_html = ""
    for rank, row in latest_sorted.head(16).iterrows():
        player = row["player"]
        prob = row["p_title"]
        prev = prev_lookup.get(player)
        if prev is not None:
            delta = prob - prev
            if abs(delta) < 0.001:
                delta_html = "<span style='color:#9E9E9E;'>—</span>"
            elif delta > 0:
                delta_html = f"<span style='color:#00703C;'>+{delta:.1%}</span>"
            else:
                delta_html = f"<span style='color:#C62828;'>{delta:.1%}</span>"
        else:
            delta_html = ""

        bar_pct = prob / latest_sorted["p_title"].iloc[0] * 100
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px; color:#9E9E9E; font-size:13px;">{rank+1}</td>
          <td style="padding:8px 4px; font-size:14px; font-weight:500;">{player}</td>
          <td style="padding:8px 12px; font-size:14px; font-weight:700; color:#00703C;">{prob:.1%}</td>
          <td style="padding:8px 12px; width:180px;">
            <div style="background:#E8F5E9; border-radius:4px; height:10px; width:100%;">
              <div style="background:#00703C; border-radius:4px; height:10px; width:{bar_pct:.1f}%;"></div>
            </div>
          </td>
          <td style="padding:8px 12px; font-size:13px; text-align:right;">{delta_html}</td>
        </tr>
        """

    delta_header = f"vs {sorted_dates[-2]}" if prev_titles is not None else ""
    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-family:sans-serif;">
      <thead>
        <tr style="border-bottom:2px solid #E0E0E0;">
          <th style="padding:8px 12px; text-align:left; color:#9E9E9E; font-size:12px; font-weight:500;">#</th>
          <th style="padding:8px 4px; text-align:left; color:#9E9E9E; font-size:12px; font-weight:500;">Player</th>
          <th style="padding:8px 12px; text-align:left; color:#9E9E9E; font-size:12px; font-weight:500;">P(Title)</th>
          <th style="padding:8px 12px; text-align:left; color:#9E9E9E; font-size:12px; font-weight:500;">Odds</th>
          <th style="padding:8px 12px; text-align:right; color:#9E9E9E; font-size:12px; font-weight:500;">{delta_header}</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Line chart only if we have multiple snapshots
    title_rows = []
    for date, snap in snapshots.items():
        if "titles" not in snap:
            continue
        for _, trow in snap["titles"].iterrows():
            title_rows.append({"Date": date, "Player": trow["player"], "P(Title)": trow["p_title"]})

    if len(snapshots) >= 2 and title_rows:
        st.markdown("---")
        st.subheader("Odds movement over time")
        title_history = pd.DataFrame(title_rows)
        top_players = latest_sorted.head(8)["player"].tolist()
        filtered = title_history[title_history["Player"].isin(top_players)]
        fig_title = px.line(
            filtered, x="Date", y="P(Title)", color="Player",
            markers=True, title="Title probability — top 8 contenders",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_title.update_layout(
            yaxis_tickformat=".1%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1A1A1A",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.4},
            height=400,
        )
        st.plotly_chart(fig_title, use_container_width=True)
    else:
        st.caption("Odds movement chart will appear once multiple daily snapshots are available.")
else:
    st.info("Title probability data not found.")
