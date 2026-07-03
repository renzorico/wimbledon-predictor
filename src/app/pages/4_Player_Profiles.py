"""Player profile page — radar charts, rolling form, surface splits."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR

st.set_page_config(page_title="Player Profiles", layout="wide")
st.title("Player profiles")

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_matches():
    return pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")

matches = load_matches()

# Build player list from recent data (2024+)
recent = matches[matches["tourney_date"].dt.year >= 2024]
players_w = recent[["winner_id", "winner_name"]].rename(
    columns={"winner_id": "id", "winner_name": "name"}
)
players_l = recent[["loser_id", "loser_name"]].rename(
    columns={"loser_id": "id", "loser_name": "name"}
)
player_list = pd.concat([players_w, players_l]).drop_duplicates("id")
player_list = player_list.sort_values("name").reset_index(drop=True)

# ── Player selector ──────────────────────────────────────────────────────
selected_name = st.selectbox(
    "Select player",
    player_list["name"].tolist(),
    index=player_list["name"].tolist().index("Jannik Sinner")
    if "Jannik Sinner" in player_list["name"].values
    else 0,
)
selected_id = player_list[player_list["name"] == selected_name]["id"].iloc[0]

# ── Compute player stats ─────────────────────────────────────────────────
won = matches[matches["winner_id"] == selected_id]
lost = matches[matches["loser_id"] == selected_id]
total = len(won) + len(lost)

# Surface breakdown
surfaces = ["Grass", "Hard", "Clay"]
surface_stats = []
for surface in surfaces:
    w = len(won[won["surface"].str.lower() == surface.lower()])
    l = len(lost[lost["surface"].str.lower() == surface.lower()])
    t = w + l
    surface_stats.append({
        "Surface": surface,
        "W": w,
        "L": l,
        "Total": t,
        "Win %": f"{w/t:.1%}" if t > 0 else "—",
    })

# Recent form (last 20 matches, all surfaces)
all_player = pd.concat([
    won[["tourney_date", "tourney_name", "surface", "loser_name", "score"]].assign(result="W"),
    lost[["tourney_date", "tourney_name", "surface", "winner_name", "score"]].rename(
        columns={"winner_name": "loser_name"}
    ).assign(result="L"),
]).sort_values("tourney_date", ascending=False).head(20)

# Serve/return stats (recent matches with stats)
recent_won = won[won["tourney_date"].dt.year >= 2024]
serve_stats = {
    "1st Serve In %": recent_won["w_1st_in_pct"].dropna().mean(),
    "1st Serve Won %": recent_won["w_1st_won_pct"].dropna().mean(),
    "2nd Serve Won %": recent_won["w_2nd_won_pct"].dropna().mean(),
    "Return Pts Won %": recent_won["w_return_pct"].dropna().mean(),
    "BP Saved %": recent_won["w_bp_saved_pct"].dropna().mean(),
    "Ace Rate": recent_won["w_ace_rate"].dropna().mean(),
}

# ── Display ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Career W-L", f"{len(won)}-{len(lost)}")
col2.metric("Win rate", f"{len(won)/total:.1%}" if total > 0 else "—")
col3.metric(
    "Grass W-L",
    f"{len(won[won['surface'].str.lower()=='grass'])}-"
    f"{len(lost[lost['surface'].str.lower()=='grass'])}",
)
col4.metric("Matches (2000–2026)", f"{total:,}")

# ── Radar chart ───────────────────────────────────────────────────────────
st.subheader("Performance radar")

categories = list(serve_stats.keys())
values = [serve_stats[c] if not np.isnan(serve_stats[c]) else 0 for c in categories]
# Normalize to 0-1 scale
max_vals = [1.0, 1.0, 1.0, 0.6, 1.0, 3.0]  # reasonable maxes
normalized = [min(v / m, 1.0) for v, m in zip(values, max_vals)]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=normalized + [normalized[0]],
    theta=categories + [categories[0]],
    fill="toself",
    fillcolor="rgba(0,112,60,0.2)",
    line_color="#00703C",
    name=selected_name,
))
fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#FAFAFA",
    showlegend=False,
    height=400,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Surface breakdown ─────────────────────────────────────────────────────
st.subheader("Surface breakdown")
st.dataframe(pd.DataFrame(surface_stats), use_container_width=True, hide_index=True)

# ── Recent form ───────────────────────────────────────────────────────────
st.subheader("Recent form (last 20 matches)")
if not all_player.empty:
    form_display = all_player[["tourney_date", "tourney_name", "surface", "result", "loser_name", "score"]].rename(
        columns={"tourney_date": "Date", "tourney_name": "Tournament", "surface": "Surface", "result": "W/L", "loser_name": "Opponent", "score": "Score"}
    )
    st.dataframe(form_display, use_container_width=True, hide_index=True)
else:
    st.info("No recent match data available.")

# ── Serve/return stats table ──────────────────────────────────────────────
st.subheader("Serve & return averages (2024+, wins only)")
stats_df = pd.DataFrame([
    {"Stat": k, "Value": f"{v:.1%}" if v < 1 else f"{v:.2f}"}
    for k, v in serve_stats.items()
    if not np.isnan(v)
])
st.dataframe(stats_df, use_container_width=True, hide_index=True)
