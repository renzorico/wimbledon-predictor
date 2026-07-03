"""Live bracket page — dynamic visual bracket with results and predictions."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR
from src.simulation.bracket import MATCHES_PER_ROUND, ROUNDS, Match, Player
from src.simulation.draw_loader import get_bracket_summary, load_wimbledon_2026_draw

st.set_page_config(page_title="Live Bracket", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
ROUND_LABELS = {
    "R1": "Round 1 (128)",
    "R2": "Round 2 (64)",
    "R3": "Round 3 (32)",
    "R4": "Round 4 (16)",
    "QF": "Quarter-Finals",
    "SF": "Semi-Finals",
    "F": "Final",
}

PREDICTIONS_DIR = DATA_DIR / "predictions"

CSS = """
<style>
.match-card {
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
    font-family: monospace;
    font-size: 13px;
}
.match-card-header {
    background: #1a1a1a;
    color: #888;
    font-size: 10px;
    padding: 2px 8px;
    letter-spacing: 0.05em;
}
.match-row {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    gap: 8px;
    background: #111;
    border-top: 1px solid #222;
}
.match-row.winner {
    background: #0a3d1a;
    color: #4ade80;
}
.match-row.loser {
    background: #1a0808;
    color: #6b3a3a;
}
.match-row.predicted-winner {
    background: #1a1a3d;
    color: #818cf8;
}
.match-row.predicted-loser {
    background: #111;
    color: #555;
}
.match-row.tbd {
    color: #555;
    font-style: italic;
}
.player-name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.seed-badge {
    color: #facc15;
    font-weight: bold;
    min-width: 28px;
    font-size: 11px;
}
.prob-bar-wrap {
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 100px;
}
.prob-label {
    font-size: 11px;
    min-width: 34px;
    text-align: right;
    color: #aaa;
}
.prob-bar-track {
    width: 60px;
    height: 6px;
    background: #222;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.prob-bar-fill.high { background: #818cf8; }
.prob-bar-fill.low  { background: #333; }
</style>
"""


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource(ttl=300)
def _load_bracket_data():
    bracket = load_wimbledon_2026_draw()
    summary = get_bracket_summary(bracket)
    return bracket, summary


@st.cache_data(ttl=300)
def _load_predictions() -> dict[str, dict]:
    """Load latest snapshot predictions keyed by match_id."""
    if not PREDICTIONS_DIR.exists():
        return {}

    date_dirs = sorted(
        [d for d in PREDICTIONS_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )
    for date_dir in date_dirs:
        pred_file = date_dir / "match_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            result: dict[str, dict] = {}
            for _, row in df.iterrows():
                result[str(row["match_id"])] = {
                    "player_a": str(row["player_a"]),
                    "player_b": str(row["player_b"]),
                    "p_a": float(row["p_player_a"]),
                    "p_b": float(row["p_player_b"]),
                    "predicted_winner": str(row["predicted_winner"]),
                    "confidence": float(row["confidence"]),
                    "snapshot_date": str(row["snapshot_date"]),
                }
            return result
    return {}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _player_label(player: Player | None) -> tuple[str, str]:
    """Return (seed_str, name_str) for display."""
    if player is None:
        return ("", "TBD")
    seed_str = f"[{player.seed}]" if player.seed else ""
    return (seed_str, player.name)


def _render_match_card(match: Match, pred: dict | None, match_label: str) -> str:
    """Return HTML for a single match card."""
    seed_a, name_a = _player_label(match.player_a)
    seed_b, name_b = _player_label(match.player_b)

    is_tbd_a = match.player_a is None
    is_tbd_b = match.player_b is None

    header = f'<div class="match-card-header">{match_label}</div>'

    # ── Locked (result known) ─────────────────────────────────────────────
    if match.locked and match.winner is not None:
        winner_name = match.winner.name

        class_a = "winner" if name_a == winner_name else "loser"
        class_b = "winner" if name_b == winner_name else "loser"

        row_a = (
            f'<div class="match-row {class_a}">'
            f'<span class="seed-badge">{seed_a}</span>'
            f'<span class="player-name">{name_a}</span>'
            f"</div>"
        )
        row_b = (
            f'<div class="match-row {class_b}">'
            f'<span class="seed-badge">{seed_b}</span>'
            f'<span class="player-name">{name_b}</span>'
            f"</div>"
        )
        return f'<div class="match-card">{header}{row_a}{row_b}</div>'

    # ── Both players TBD ──────────────────────────────────────────────────
    if is_tbd_a and is_tbd_b:
        row_a = '<div class="match-row tbd"><span class="player-name">TBD</span></div>'
        row_b = '<div class="match-row tbd"><span class="player-name">TBD</span></div>'
        return f'<div class="match-card">{header}{row_a}{row_b}</div>'

    # ── Prediction available ───────────────────────────────────────────────
    if pred is not None and not is_tbd_a and not is_tbd_b:
        p_a = pred["p_a"]
        p_b = pred["p_b"]
        predicted_winner = pred["predicted_winner"]

        class_a = "predicted-winner" if name_a == predicted_winner else "predicted-loser"
        class_b = "predicted-winner" if name_b == predicted_winner else "predicted-loser"

        fill_a = "high" if name_a == predicted_winner else "low"
        fill_b = "high" if name_b == predicted_winner else "low"

        prob_bar_a = (
            f'<div class="prob-bar-wrap">'
            f'<span class="prob-label">{p_a:.0%}</span>'
            f'<div class="prob-bar-track">'
            f'<div class="prob-bar-fill {fill_a}" style="width:{p_a*100:.0f}%"></div>'
            f"</div></div>"
        )
        prob_bar_b = (
            f'<div class="prob-bar-wrap">'
            f'<span class="prob-label">{p_b:.0%}</span>'
            f'<div class="prob-bar-track">'
            f'<div class="prob-bar-fill {fill_b}" style="width:{p_b*100:.0f}%"></div>'
            f"</div></div>"
        )

        row_a = (
            f'<div class="match-row {class_a}">'
            f'<span class="seed-badge">{seed_a}</span>'
            f'<span class="player-name">{name_a}</span>'
            f"{prob_bar_a}</div>"
        )
        row_b = (
            f'<div class="match-row {class_b}">'
            f'<span class="seed-badge">{seed_b}</span>'
            f'<span class="player-name">{name_b}</span>'
            f"{prob_bar_b}</div>"
        )
        return f'<div class="match-card">{header}{row_a}{row_b}</div>'

    # ── One player known, one TBD (no prediction yet) ─────────────────────
    class_a = "tbd" if is_tbd_a else "match-row"
    class_b = "tbd" if is_tbd_b else "match-row"

    row_a = (
        f'<div class="match-row {class_a}">'
        f'<span class="seed-badge">{seed_a}</span>'
        f'<span class="player-name">{name_a}</span>'
        f"</div>"
    )
    row_b = (
        f'<div class="match-row {class_b}">'
        f'<span class="seed-badge">{seed_b}</span>'
        f'<span class="player-name">{name_b}</span>'
        f"</div>"
    )
    return f'<div class="match-card">{header}{row_a}{row_b}</div>'


def _render_round_grid(
    matches: list[Match],
    predictions: dict[str, dict],
    cols_per_row: int = 2,
) -> None:
    """Render a grid of match cards for a round."""
    sorted_matches = sorted(matches, key=lambda m: int(m.match_id.split("_M")[1]))
    rows = [sorted_matches[i : i + cols_per_row] for i in range(0, len(sorted_matches), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col_idx, match in enumerate(row):
            pred = predictions.get(match.match_id)
            round_num = int(match.match_id.split("_M")[1])
            label = f"{match.round_name}  Match {round_num}"
            html = _render_match_card(match, pred, label)
            with cols[col_idx]:
                st.markdown(html, unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────────────────────

st.markdown(CSS, unsafe_allow_html=True)
st.title("Live Bracket")

bracket, summary = _load_bracket_data()
predictions = _load_predictions()

snapshot_dates = sorted({v["snapshot_date"] for v in predictions.values()}, reverse=True)
snapshot_label = snapshot_dates[0] if snapshot_dates else "none"

# ── Tournament progress ───────────────────────────────────────────────────────
total_matches = sum(MATCHES_PER_ROUND.values())  # 127
locked_total = summary["locked_total"]
progress_pct = locked_total / total_matches

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches completed", locked_total, delta=f"of {total_matches}")
c2.metric("R1 complete", summary["r1_complete"], delta=f"of 64")
c3.metric("R1 remaining", summary["r1_remaining"])
c4.metric("Prediction snapshot", snapshot_label)

st.progress(progress_pct, text=f"Tournament progress: {progress_pct:.0%}")
st.divider()

# ── Round selector tabs ───────────────────────────────────────────────────────
tab_labels = [ROUND_LABELS[r] for r in ROUNDS]
tabs = st.tabs(tab_labels)

for tab, round_name in zip(tabs, ROUNDS):
    with tab:
        round_matches = bracket.get_round_matches(round_name)

        if not round_matches:
            st.info("No matches found for this round yet.")
            continue

        total_in_round = MATCHES_PER_ROUND[round_name]
        locked_in_round = sum(1 for m in round_matches if m.locked)
        predicted_in_round = sum(1 for m in round_matches if m.match_id in predictions)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Completed", f"{locked_in_round} / {total_in_round}")
        mc2.metric("Predicted (model)", predicted_in_round)
        mc3.metric("TBD", total_in_round - locked_in_round - len(round_matches) + locked_in_round)

        # Columns per row: fewer for later rounds to give cards more room
        if round_name in ("R1", "R2"):
            cols_per_row = 4
        elif round_name in ("R3", "R4"):
            cols_per_row = 3
        elif round_name == "QF":
            cols_per_row = 2
        else:
            cols_per_row = 1

        _render_round_grid(round_matches, predictions, cols_per_row=cols_per_row)

# ── Late-round visual summary ─────────────────────────────────────────────────
st.divider()
st.subheader("Final stages at a glance")

qf_matches = sorted(bracket.get_round_matches("QF"), key=lambda m: int(m.match_id.split("_M")[1]))
sf_matches = sorted(bracket.get_round_matches("SF"), key=lambda m: int(m.match_id.split("_M")[1]))
f_matches = bracket.get_round_matches("F")

# QF -> SF -> F in three column groups
col_left, col_mid_left, col_center, col_mid_right, col_right = st.columns([3, 1, 3, 1, 3])

def _render_compact(match: Match, pred: dict | None) -> str:
    if match is None:
        return ""
    label = match.match_id
    return _render_match_card(match, pred, label)


with col_left:
    st.caption("Quarter-Finals (top half)")
    for m in qf_matches[:2]:
        st.markdown(_render_compact(m, predictions.get(m.match_id)), unsafe_allow_html=True)

with col_mid_left:
    st.write("")

with col_center:
    st.caption("Semi-Finals")
    for m in sf_matches:
        st.markdown(_render_compact(m, predictions.get(m.match_id)), unsafe_allow_html=True)
    st.caption("Final")
    for m in f_matches:
        st.markdown(_render_compact(m, predictions.get(m.match_id)), unsafe_allow_html=True)

with col_mid_right:
    st.write("")

with col_right:
    st.caption("Quarter-Finals (bottom half)")
    for m in qf_matches[2:]:
        st.markdown(_render_compact(m, predictions.get(m.match_id)), unsafe_allow_html=True)

# ── Champion readout ──────────────────────────────────────────────────────────
champion = bracket.get_champion()
if champion:
    seed_str = f" (seed {champion.seed})" if champion.seed else ""
    st.success(f"Champion: {champion.name}{seed_str}")
