"""Prediction Tracker — accuracy, match outcomes, and title odds over time."""

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

st.set_page_config(page_title="Prediction Tracker", layout="wide")
st.title("Prediction tracker")
st.caption("Every prediction the model made — and how it went.")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_snapshots() -> dict[str, dict]:
    if not PREDICTIONS_DIR.exists():
        return {}
    snapshots = {}
    for d in sorted(PREDICTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        snap: dict = {"date": d.name}
        for fname, key in [
            ("accuracy_summary.json", "summary"),
            ("metadata.json", "metadata"),
        ]:
            p = d / fname
            if p.exists():
                snap[key] = json.loads(p.read_text())
        for fname, key in [
            ("match_predictions_evaluated.csv", "evaluated"),
            ("title_probabilities.csv", "titles"),
        ]:
            p = d / fname
            if p.exists():
                snap[key] = pd.read_csv(p)
        snapshots[d.name] = snap
    return snapshots


snapshots = load_snapshots()

if not snapshots:
    st.warning("No prediction snapshots found in `data/predictions/`.")
    st.stop()

sorted_dates = sorted(snapshots.keys())
latest_date = sorted_dates[-1]


# ── Section 1: Accuracy summary ───────────────────────────────────────────────

st.subheader("Accuracy summary")

accuracy_rows = []
for date, snap in snapshots.items():
    s = snap.get("summary", {})
    accuracy_rows.append({
        "Date": date,
        "Predictions made": s.get("predictions", 0),
        "Resolved": s.get("resolved", 0),
        "Correct": s.get("correct", 0),
        "Accuracy": s.get("accuracy"),
        "Avg confidence": s.get("average_confidence_resolved"),
    })
acc_df = pd.DataFrame(accuracy_rows)
total_resolved = int(acc_df["Resolved"].sum())
total_correct = int(acc_df["Correct"].sum())
overall_accuracy = total_correct / total_resolved if total_resolved > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Snapshots saved", len(snapshots))
c2.metric("Total predictions", int(acc_df["Predictions made"].sum()))
c3.metric("Resolved", total_resolved)
c4.metric(
    "Overall accuracy",
    f"{overall_accuracy:.1%}" if overall_accuracy is not None else "—",
)

if total_resolved > 0:
    col_l, col_r = st.columns(2)
    with col_l:
        resolved_acc = acc_df[acc_df["Accuracy"].notna()]
        fig_acc = px.line(
            resolved_acc, x="Date", y="Accuracy", markers=True,
            title="Daily accuracy",
            color_discrete_sequence=["#00703C"],
        )
        fig_acc.add_hline(y=0.5, line_dash="dash", line_color="#CCCCCC",
                          annotation_text="Baseline 50%", annotation_position="right")
        fig_acc.update_layout(
            yaxis_range=[0, 1], yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1A1A1A", height=280, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    with col_r:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=resolved_acc["Date"], y=resolved_acc["Correct"],
            name="Correct", marker_color="#00703C",
        ))
        fig_bar.add_trace(go.Bar(
            x=resolved_acc["Date"],
            y=resolved_acc["Resolved"] - resolved_acc["Correct"],
            name="Incorrect", marker_color="#C62828",
        ))
        fig_bar.update_layout(
            barmode="stack", title="Results by day",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1A1A1A", height=280, margin=dict(t=40, b=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Accuracy charts will appear once matches are played and predictions are resolved.")

with st.expander("Snapshot summary table"):
    st.dataframe(
        acc_df.style.format({
            "Accuracy": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
            "Avg confidence": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ── Section 2: Match prediction record ───────────────────────────────────────

st.markdown("---")
st.subheader("Match prediction record")
st.caption("Every match the model predicted — when, what it said, and what happened.")

all_preds = pd.concat(
    [snap["evaluated"] for snap in snapshots.values() if "evaluated" in snap],
    ignore_index=True,
)

if all_preds.empty:
    st.info("No predictions recorded yet.")
    st.stop()

ROUND_ORDER = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "QF": 5, "SF": 6, "F": 7}
all_preds = all_preds.copy()
all_preds["_round_order"] = all_preds["round"].map(ROUND_ORDER).fillna(9)
all_preds = all_preds.sort_values(
    ["_round_order", "snapshot_date", "match_id"]
).reset_index(drop=True)

view_filter = st.radio(
    "Filter",
    ["All", "Correct", "Incorrect", "Pending"],
    horizontal=True,
)

if view_filter == "Correct":
    view_df = all_preds[all_preds["correct"] == True]
elif view_filter == "Incorrect":
    view_df = all_preds[
        (all_preds["resolved"] == True) & (all_preds["correct"] == False)
    ]
elif view_filter == "Pending":
    view_df = all_preds[all_preds["resolved"] == False]
else:
    view_df = all_preds

if view_df.empty:
    st.info("No predictions match this filter.")
else:
    ROUND_LABELS = {
        "R1": "Round 1", "R2": "Round 2", "R3": "Round 3", "R4": "Round 4 (R16)",
        "QF": "Quarter-Finals", "SF": "Semi-Finals", "F": "Final",
    }
    for rnd, group in view_df.groupby("round", sort=False):
        resolved_n = int(group["resolved"].sum())
        correct_n = int(group["correct"].sum())
        rnd_label = ROUND_LABELS.get(rnd, rnd)
        if resolved_n > 0:
            acc_label = f"{correct_n}/{resolved_n} correct"
        else:
            acc_label = f"{len(group)} pending"

        st.markdown(
            f"<p style='font-weight:700; color:#4B2D83; font-size:15px; "
            f"margin:1.2rem 0 0.5rem;'>{rnd_label} "
            f"<span style='font-weight:400; color:#9E9E9E; font-size:13px;'>"
            f"— {acc_label}</span></p>",
            unsafe_allow_html=True,
        )

        cols = st.columns(2)
        for i, (_, row) in enumerate(group.iterrows()):
            p_a = float(row["p_player_a"])
            p_b = float(row["p_player_b"])
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            pred_winner = str(row["predicted_winner"])
            resolved = bool(row.get("resolved", False))
            correct = bool(row.get("correct", False))
            actual = row.get("actual_winner", None)
            snap_date = str(row.get("snapshot_date", ""))

            if resolved and correct:
                border = "#00703C"
                bg = "#F1F8F4"
                status = f"Correct — {actual} won"
                status_color = "#00703C"
            elif resolved and not correct:
                border = "#C62828"
                bg = "#FFF5F5"
                status = f"Incorrect — {actual} won"
                status_color = "#C62828"
            else:
                border = "#E0E0E0"
                bg = "#FAFAFA"
                status = "Prediction pending"
                status_color = "#9E9E9E"

            a_weight = "700" if pred_winner == player_a else "400"
            b_weight = "700" if pred_winner == player_b else "400"
            a_color = "#00703C" if pred_winner == player_a else "#333"
            b_color = "#00703C" if pred_winner == player_b else "#333"

            with cols[i % 2]:
                st.markdown(
                    f'<div style="border:1.5px solid {border}; border-radius:8px; '
                    f'padding:10px 14px; margin-bottom:8px; background:{bg}; '
                    f'font-family:sans-serif;">'
                    f'<div style="font-size:10px; color:#AAAAAA; margin-bottom:6px; '
                    f'text-transform:uppercase; letter-spacing:0.05em;">'
                    f'Predicted {snap_date}</div>'
                    f'<div style="display:flex; justify-content:space-between; '
                    f'align-items:center; margin-bottom:4px;">'
                    f'<span style="color:{a_color}; font-size:14px; '
                    f'font-weight:{a_weight};">{player_a}</span>'
                    f'<span style="font-size:12px; color:#888;">{p_a:.0%}</span></div>'
                    f'<div style="width:100%; height:5px; background:#E8E8E8; '
                    f'border-radius:3px; margin-bottom:4px;">'
                    f'<div style="width:{p_a*100:.1f}%; height:100%; background:#00703C; '
                    f'border-radius:3px;"></div></div>'
                    f'<div style="display:flex; justify-content:space-between; '
                    f'align-items:center; margin-bottom:6px;">'
                    f'<span style="color:{b_color}; font-size:14px; '
                    f'font-weight:{b_weight};">{player_b}</span>'
                    f'<span style="font-size:12px; color:#888;">{p_b:.0%}</span></div>'
                    f'<div style="font-size:11px; color:{status_color}; '
                    f'border-top:1px solid #EFEFEF; padding-top:5px;">'
                    f'{status}'
                    f'<span style="float:right; color:#AAAAAA;">'
                    f'Confidence {max(p_a, p_b):.0%}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Section 3: Title odds leaderboard ────────────────────────────────────────

st.markdown("---")
st.subheader("Title odds leaderboard")

latest_titles = snapshots[latest_date].get("titles")

if latest_titles is not None:
    latest_sorted = latest_titles.sort_values("p_title", ascending=False).reset_index(drop=True)

    prev_lookup: dict[str, float] = {}
    if len(sorted_dates) >= 2:
        prev_snap = snapshots.get(sorted_dates[-2], {})
        prev_titles = prev_snap.get("titles")
        if prev_titles is not None:
            prev_lookup = dict(zip(prev_titles["player"], prev_titles["p_title"]))

    top16 = latest_sorted.head(16).copy()

    fig_odds = go.Figure(go.Bar(
        x=top16["p_title"] * 100,
        y=top16["player"],
        orientation="h",
        marker_color="#00703C",
        text=[f"{v:.1f}%" for v in top16["p_title"] * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig_odds.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1A1A1A",
        xaxis=dict(
            title="P(Title) %",
            range=[0, top16["p_title"].max() * 120],
            showgrid=False,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        margin=dict(l=10, r=70, t=10, b=20),
        height=max(320, len(top16) * 28),
        showlegend=False,
    )
    st.plotly_chart(fig_odds, use_container_width=True)

    if prev_lookup:
        delta_rows = []
        for _, row in top16.iterrows():
            player = row["player"]
            prob = row["p_title"]
            prev = prev_lookup.get(player)
            delta = prob - prev if prev is not None else None
            delta_rows.append({
                "Player": player,
                "P(Title)": prob,
                "Change vs previous": delta,
            })
        delta_df = pd.DataFrame(delta_rows)
        st.caption(f"vs snapshot {sorted_dates[-2]}")
        st.dataframe(
            delta_df.style.format({
                "P(Title)": "{:.2%}",
                "Change vs previous": lambda v: (
                    f"+{v:.2%}" if v is not None and v > 0
                    else (f"{v:.2%}" if v is not None else "—")
                ),
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Odds movement — only when 2+ snapshots exist
    title_rows = []
    for date, snap in snapshots.items():
        if "titles" not in snap:
            continue
        for _, trow in snap["titles"].iterrows():
            title_rows.append({
                "Date": date,
                "Player": trow["player"],
                "P(Title)": trow["p_title"],
            })

    if len(snapshots) >= 2 and title_rows:
        st.markdown("---")
        st.subheader("Odds movement over time")
        title_history = pd.DataFrame(title_rows)
        top_players = latest_sorted.head(8)["player"].tolist()
        selected = st.multiselect(
            "Players to compare",
            sorted(title_history["Player"].unique()),
            default=top_players,
        )
        if selected:
            filtered = title_history[title_history["Player"].isin(selected)]
            fig_mv = px.line(
                filtered, x="Date", y="P(Title)", color="Player",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_mv.update_layout(
                yaxis_tickformat=".1%",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#1A1A1A",
                legend=dict(orientation="h", y=-0.3),
                height=380,
            )
            st.plotly_chart(fig_mv, use_container_width=True)
    else:
        st.caption(
            "Odds movement chart will appear once multiple daily snapshots are available."
        )
else:
    st.info("Title probability data not found.")
