"""Live bracket page — visual tournament bracket with Plotly finals path and quarter views."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR
from src.simulation.bracket import MATCHES_PER_ROUND, ROUNDS, Match, Player
from src.simulation.draw_loader import get_bracket_summary, load_wimbledon_2026_draw

st.set_page_config(page_title="Live Bracket", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────

PREDICTIONS_DIR = DATA_DIR / "predictions"

# Bracket figure geometry
BOX_W = 3.8
PLAYER_H = 0.65
PLAYER_PAD = 0.05

POSITIONS = {
    "QF_M1": 8.5,
    "QF_M2": 3.5,
    "SF_M1": 6.0,
    "QF_M3": -3.5,
    "QF_M4": -8.5,
    "SF_M2": -6.0,
    "F_M1": 0.0,
}

ROUND_X = {"QF": 0.0, "SF": 5.5, "F": 11.0}

# Colors — light theme
COLOR_LOCKED_WIN_FILL = "#e8f5e9"
COLOR_LOCKED_WIN_LINE = "#00703C"
COLOR_LOCKED_WIN_TEXT = "#1B5E20"
COLOR_LOCKED_LOSE_FILL = "#ffebee"
COLOR_LOCKED_LOSE_LINE = "#ffcdd2"
COLOR_LOCKED_LOSE_TEXT = "#B71C1C"
COLOR_PRED_WIN_FILL = "#EDE7F6"
COLOR_PRED_WIN_LINE = "#4B2D83"
COLOR_PRED_WIN_TEXT = "#4B2D83"
COLOR_PRED_LOSE_FILL = "#FAFAFA"
COLOR_PRED_LOSE_LINE = "#E0E0E0"
COLOR_PRED_LOSE_TEXT = "#9E9E9E"
COLOR_TBD_FILL = "#F5F5F5"
COLOR_TBD_LINE = "#E0E0E0"
COLOR_TBD_TEXT = "#BDBDBD"
COLOR_CONNECTOR = "#BBBBBB"
COLOR_SEED = "#B8860B"
COLOR_ROUND_HEADER = "#00703C"

# Quarter draw ranges: (r1_start, r1_end) by quarter index (0-based)
QUARTER_R1_RANGES = [
    (1, 16),   # Q1
    (17, 32),  # Q2
    (33, 48),  # Q3
    (49, 64),  # Q4
]

CSS = """
<style>
.match-card {
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #E0E0E0;
}
.match-card-header {
    background: #F5F5F5;
    color: #9E9E9E;
    font-size: 10px;
    padding: 3px 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid #E8E8E8;
}
.match-row {
    display: flex;
    align-items: center;
    padding: 7px 10px;
    gap: 8px;
    background: #FFFFFF;
    border-top: 1px solid #F0F0F0;
    min-height: 34px;
}
.match-row.winner {
    background: #e8f5e9;
    color: #1B5E20;
}
.match-row.loser {
    background: #ffebee;
    color: #B71C1C;
}
.match-row.predicted-winner {
    background: #EDE7F6;
    color: #4B2D83;
}
.match-row.predicted-loser {
    background: #FAFAFA;
    color: #757575;
}
.match-row.tbd {
    color: #BDBDBD;
    font-style: italic;
}
.player-name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.seed-badge {
    color: #B8860B;
    font-weight: 700;
    min-width: 28px;
    font-size: 11px;
}
.prob-label {
    font-size: 11px;
    min-width: 38px;
    text-align: right;
    color: #888;
    font-variant-numeric: tabular-nums;
}
</style>
"""


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_resource(ttl=300)
def _load_bracket():
    bracket = load_wimbledon_2026_draw()
    summary = get_bracket_summary(bracket)
    return bracket, summary


@st.cache_data(ttl=300)
def _load_predictions() -> tuple[dict[str, dict], str]:
    """Load latest snapshot predictions. Returns (predictions_dict, snapshot_date)."""
    if not PREDICTIONS_DIR.exists():
        return {}, "none"

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
                }
            return result, date_dir.name
    return {}, "none"


# ── Name helpers ───────────────────────────────────────────────────────────────

def _abbrev_name(name: str, max_len: int = 16) -> str:
    """Abbreviate first name to initial if full name exceeds max_len."""
    if len(name) <= max_len:
        return name
    parts = name.strip().split()
    if len(parts) <= 1:
        return name[:max_len]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"[:max_len]


def _player_display(player: Player | None) -> tuple[str, str]:
    """Return (seed_str, abbreviated_name) for a player."""
    if player is None:
        return ("", "TBD")
    seed_str = f"[{player.seed}]" if player.seed else ""
    return (seed_str, _abbrev_name(player.name))


# ── Plotly bracket builder ─────────────────────────────────────────────────────

def _row_bounds(y_center: float, slot: str) -> tuple[float, float]:
    """Return (y0, y1) for a player slot ('a' = top, 'b' = bottom)."""
    if slot == "a":
        return y_center + PLAYER_PAD, y_center + PLAYER_PAD + PLAYER_H
    return y_center - PLAYER_PAD - PLAYER_H, y_center - PLAYER_PAD


def _row_center(y_center: float, slot: str) -> float:
    y0, y1 = _row_bounds(y_center, slot)
    return (y0 + y1) / 2


def _match_colors(
    player: Player | None,
    match: Match,
    is_slot_a: bool,
    pred: dict | None,
) -> tuple[str, str, str]:
    """Return (fill_color, line_color, text_color) for a player row."""
    if player is None:
        return COLOR_TBD_FILL, COLOR_TBD_LINE, COLOR_TBD_TEXT

    if match.locked and match.winner is not None:
        is_winner = match.winner.name == player.name
        if is_winner:
            return COLOR_LOCKED_WIN_FILL, COLOR_LOCKED_WIN_LINE, COLOR_LOCKED_WIN_TEXT
        return COLOR_LOCKED_LOSE_FILL, COLOR_LOCKED_LOSE_LINE, COLOR_LOCKED_LOSE_TEXT

    if pred is not None:
        is_predicted_winner = pred["predicted_winner"] == player.name
        if is_predicted_winner:
            return COLOR_PRED_WIN_FILL, COLOR_PRED_WIN_LINE, COLOR_PRED_WIN_TEXT
        return COLOR_PRED_LOSE_FILL, COLOR_PRED_LOSE_LINE, COLOR_PRED_LOSE_TEXT

    return COLOR_TBD_FILL, COLOR_TBD_LINE, COLOR_TBD_TEXT


def _add_match_box(
    fig: go.Figure,
    match_id: str,
    match: Match,
    pred: dict | None,
    x_left: float,
) -> None:
    """Add rectangles and text annotations for one match box."""
    y_center = POSITIONS[match_id]
    x_right = x_left + BOX_W

    for slot, player in [("a", match.player_a), ("b", match.player_b)]:
        y0, y1 = _row_bounds(y_center, slot)
        yc = (y0 + y1) / 2
        fill, line_col, text_col = _match_colors(player, match, slot == "a", pred)

        fig.add_shape(
            type="rect",
            x0=x_left, x1=x_right,
            y0=y0, y1=y1,
            fillcolor=fill,
            line=dict(color=line_col, width=1),
            layer="below",
        )

        seed_str, name_str = _player_display(player)
        label = f"{seed_str} {name_str}".strip() if seed_str else name_str

        fig.add_annotation(
            x=x_left + 0.12, y=yc,
            text=label,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font=dict(size=11, color=text_col),
        )

        # Probability label for predictions (right-aligned)
        if pred is not None and not match.locked and player is not None:
            p_val = pred["p_a"] if slot == "a" else pred["p_b"]
            fig.add_annotation(
                x=x_right - 0.12, y=yc,
                text=f"{p_val:.0%}",
                xanchor="right", yanchor="middle",
                showarrow=False,
                font=dict(size=10, color=text_col),
            )


def _add_connector(
    fig: go.Figure,
    x_from: float,
    y_from: float,
    x_to: float,
    y_to: float,
    mid_x: float,
) -> None:
    """Add an L-shaped elbow connector (3 line segments)."""
    coords = [
        (x_from, y_from, mid_x, y_from),
        (mid_x, y_from, mid_x, y_to),
        (mid_x, y_to, x_to, y_to),
    ]
    for x0, y0, x1, y1 in coords:
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=COLOR_CONNECTOR, width=1.5),
        )


def build_finals_bracket(
    bracket_obj,
    predictions: dict[str, dict],
) -> go.Figure:
    """Build Plotly figure for QF → SF → Final visual bracket."""
    fig = go.Figure()

    mid_qf_sf = (ROUND_X["QF"] + BOX_W + ROUND_X["SF"]) / 2   # 4.65
    mid_sf_f = (ROUND_X["SF"] + BOX_W + ROUND_X["F"]) / 2      # 10.15

    # Draw match boxes
    draw_order = [
        ("QF_M1", "QF"), ("QF_M2", "QF"),
        ("QF_M3", "QF"), ("QF_M4", "QF"),
        ("SF_M1", "SF"), ("SF_M2", "SF"),
        ("F_M1", "F"),
    ]

    for match_key, round_key in draw_order:
        match = bracket_obj.matches.get(match_key)
        if match is None:
            # Create empty placeholder match for rendering
            from src.simulation.bracket import Match as BMatch
            match = BMatch(match_id=match_key, round_name=round_key)
        pred = predictions.get(match_key)
        _add_match_box(fig, match_key, match, pred, ROUND_X[round_key])

    # Connector lines: QF -> SF
    # QF_M1 (y=8.5) -> SF_M1 player_a slot (y center 6.0)
    _add_connector(fig,
        ROUND_X["QF"] + BOX_W, POSITIONS["QF_M1"],
        ROUND_X["SF"], _row_center(POSITIONS["SF_M1"], "a"),
        mid_qf_sf,
    )
    # QF_M2 (y=3.5) -> SF_M1 player_b slot
    _add_connector(fig,
        ROUND_X["QF"] + BOX_W, POSITIONS["QF_M2"],
        ROUND_X["SF"], _row_center(POSITIONS["SF_M1"], "b"),
        mid_qf_sf,
    )
    # QF_M3 (y=-3.5) -> SF_M2 player_a slot
    _add_connector(fig,
        ROUND_X["QF"] + BOX_W, POSITIONS["QF_M3"],
        ROUND_X["SF"], _row_center(POSITIONS["SF_M2"], "a"),
        mid_qf_sf,
    )
    # QF_M4 (y=-8.5) -> SF_M2 player_b slot
    _add_connector(fig,
        ROUND_X["QF"] + BOX_W, POSITIONS["QF_M4"],
        ROUND_X["SF"], _row_center(POSITIONS["SF_M2"], "b"),
        mid_qf_sf,
    )

    # Connector lines: SF -> F
    _add_connector(fig,
        ROUND_X["SF"] + BOX_W, POSITIONS["SF_M1"],
        ROUND_X["F"], _row_center(POSITIONS["F_M1"], "a"),
        mid_sf_f,
    )
    _add_connector(fig,
        ROUND_X["SF"] + BOX_W, POSITIONS["SF_M2"],
        ROUND_X["F"], _row_center(POSITIONS["F_M1"], "b"),
        mid_sf_f,
    )

    # Round header annotations
    headers = [
        ("Quarter-Finals", ROUND_X["QF"] + BOX_W / 2),
        ("Semi-Finals", ROUND_X["SF"] + BOX_W / 2),
        ("Final", ROUND_X["F"] + BOX_W / 2),
    ]
    for label, x_center in headers:
        fig.add_annotation(
            x=x_center, y=10.8,
            text=f"<b>{label}</b>",
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=13, color=COLOR_ROUND_HEADER),
        )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            range=[-0.5, 15.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[-11, 11.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        height=680,
        margin=dict(l=10, r=50, t=40, b=10),
        showlegend=False,
        font=dict(color="#1A1A1A"),
    )

    return fig


# ── Match card HTML ────────────────────────────────────────────────────────────

def _render_match_card(match: Match, pred: dict | None, match_label: str) -> str:
    """Return HTML for a single match card (light theme)."""
    is_tbd_a = match.player_a is None
    is_tbd_b = match.player_b is None
    seed_a, name_a = _player_display(match.player_a)
    seed_b, name_b = _player_display(match.player_b)

    header = f'<div class="match-card-header">{match_label}</div>'

    def _seed_span(s: str) -> str:
        if not s:
            return '<span class="seed-badge"></span>'
        return f'<span class="seed-badge">{s}</span>'

    # Locked result
    if match.locked and match.winner is not None:
        winner_name = match.winner.name
        class_a = "winner" if (match.player_a and match.player_a.name == winner_name) else "loser"
        class_b = "winner" if (match.player_b and match.player_b.name == winner_name) else "loser"
        row_a = (
            f'<div class="match-row {class_a}">'
            f'{_seed_span(seed_a)}'
            f'<span class="player-name">{name_a}</span>'
            f"</div>"
        )
        row_b = (
            f'<div class="match-row {class_b}">'
            f'{_seed_span(seed_b)}'
            f'<span class="player-name">{name_b}</span>'
            f"</div>"
        )
        return f'<div class="match-card">{header}{row_a}{row_b}</div>'

    # Both TBD
    if is_tbd_a and is_tbd_b:
        row = '<div class="match-row tbd"><span class="player-name">TBD</span></div>'
        return f'<div class="match-card">{header}{row}{row}</div>'

    # Prediction available
    if pred is not None and not is_tbd_a and not is_tbd_b:
        pw = pred["predicted_winner"]
        p_a, p_b = pred["p_a"], pred["p_b"]
        class_a = "predicted-winner" if name_a == _abbrev_name(pw) or match.player_a.name == pw else "predicted-loser"
        class_b = "predicted-winner" if name_b == _abbrev_name(pw) or match.player_b.name == pw else "predicted-loser"
        row_a = (
            f'<div class="match-row {class_a}">'
            f'{_seed_span(seed_a)}'
            f'<span class="player-name">{name_a}</span>'
            f'<span class="prob-label">{p_a:.0%}</span>'
            f"</div>"
        )
        row_b = (
            f'<div class="match-row {class_b}">'
            f'{_seed_span(seed_b)}'
            f'<span class="player-name">{name_b}</span>'
            f'<span class="prob-label">{p_b:.0%}</span>'
            f"</div>"
        )
        return f'<div class="match-card">{header}{row_a}{row_b}</div>'

    # Partial (one player known)
    class_a = "tbd" if is_tbd_a else ""
    class_b = "tbd" if is_tbd_b else ""
    row_a = (
        f'<div class="match-row {class_a}">'
        f'{_seed_span(seed_a)}'
        f'<span class="player-name">{name_a}</span>'
        f"</div>"
    )
    row_b = (
        f'<div class="match-row {class_b}">'
        f'{_seed_span(seed_b)}'
        f'<span class="player-name">{name_b}</span>'
        f"</div>"
    )
    return f'<div class="match-card">{header}{row_a}{row_b}</div>'


# ── Quarter render ─────────────────────────────────────────────────────────────

def _quarter_match_ids(quarter: int) -> dict[str, list[str]]:
    """Return {round_name: [match_ids]} for a given quarter (1-4)."""
    q = quarter - 1
    r1_start, r1_end = QUARTER_R1_RANGES[q]

    def ids(prefix: str, start: int, end: int) -> list[str]:
        return [f"{prefix}_M{i}" for i in range(start, end + 1)]

    r1_ids = ids("R1", r1_start, r1_end)
    r2_start = (r1_start - 1) // 2 + 1
    r2_end = r1_end // 2
    r2_ids = ids("R2", r2_start, r2_end)
    r3_start = (r2_start - 1) // 2 + 1
    r3_end = r2_end // 2
    r3_ids = ids("R3", r3_start, r3_end)
    r4_start = (r3_start - 1) // 2 + 1
    r4_end = r3_end // 2
    r4_ids = ids("R4", r4_start, r4_end)
    qf_num = quarter
    qf_ids = [f"QF_M{qf_num}"]

    return {
        "R1": r1_ids,
        "R2": r2_ids,
        "R3": r3_ids,
        "R4": r4_ids,
        "QF": qf_ids,
    }


def _render_round_section(
    round_name: str,
    match_ids: list[str],
    bracket_obj,
    predictions: dict[str, dict],
    cols_per_row: int,
) -> None:
    round_labels = {
        "R1": "Round 1",
        "R2": "Round 2",
        "R3": "Round 3",
        "R4": "Round 4 (R16)",
        "QF": "Quarter-Final",
    }
    st.markdown(f"**{round_labels.get(round_name, round_name)}**")

    matches = []
    for mid in match_ids:
        m = bracket_obj.matches.get(mid)
        if m is None:
            from src.simulation.bracket import Match as BMatch
            m = BMatch(match_id=mid, round_name=round_name)
        matches.append(m)

    rows = [matches[i: i + cols_per_row] for i in range(0, len(matches), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col_idx, match in enumerate(row):
            pred = predictions.get(match.match_id)
            num = match.match_id.split("_M")[1]
            label = f"Match {num}"
            html = _render_match_card(match, pred, label)
            with cols[col_idx]:
                st.markdown(html, unsafe_allow_html=True)


def _render_quarter(
    bracket_obj,
    predictions: dict[str, dict],
    quarter: int,
) -> None:
    all_match_ids = _quarter_match_ids(quarter)

    # Count remaining players (players who haven't lost yet in this quarter)
    r1_ids = all_match_ids["R1"]
    r1_matches = [bracket_obj.matches.get(mid) for mid in r1_ids]
    total_r1 = len(r1_ids) * 2  # 32 players
    losers = sum(
        1 for m in r1_matches
        if m is not None and m.locked and m.winner is not None
    )
    remaining = total_r1 - losers

    # Count remaining across all rounds in this quarter
    all_ids_flat = [mid for ids in all_match_ids.values() for mid in ids]
    locked_here = sum(
        1 for mid in all_ids_flat
        if bracket_obj.matches.get(mid) is not None and bracket_obj.matches[mid].locked
    )

    col_a, col_b = st.columns([1, 3])
    col_a.metric("Players remaining", remaining, delta=f"of {total_r1}")
    col_b.metric("Matches completed", locked_here, delta=f"of {len(all_ids_flat)}")

    st.divider()

    for round_name, match_ids in all_match_ids.items():
        cols_per_row = 2 if round_name in ("R1", "R2") else 1
        _render_round_section(round_name, match_ids, bracket_obj, predictions, cols_per_row)
        st.write("")


# ── Main page ──────────────────────────────────────────────────────────────────

st.markdown(CSS, unsafe_allow_html=True)
st.title("Live Bracket")

bracket, summary = _load_bracket()
predictions, snapshot_date = _load_predictions()

# Progress metrics
total_matches = sum(MATCHES_PER_ROUND.values())  # 127
locked_total = summary["locked_total"]
progress_pct = locked_total / total_matches

# Per-round locked counts for metrics
def _locked_in_round(rname: str) -> int:
    return sum(1 for m in bracket.matches.values() if m.round_name == rname and m.locked)

r2_locked = _locked_in_round("R2")
r3_locked = _locked_in_round("R3")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completed", f"{locked_total} / {total_matches}")
c2.metric("R1", f"{summary['r1_complete']} / 64")
c3.metric("R2", f"{r2_locked} / 32")
c4.metric("Prediction date", snapshot_date)

st.progress(progress_pct, text=f"Tournament progress: {progress_pct:.0%}")

champion = bracket.get_champion()
if champion:
    seed_label = f" (seed {champion.seed})" if champion.seed else ""
    st.success(f"Champion: {champion.name}{seed_label}")

st.divider()

# Tabs
tab_finals, tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs([
    "Finals Path (QF → Final)",
    "Quarter 1",
    "Quarter 2",
    "Quarter 3",
    "Quarter 4",
])

with tab_finals:
    fig = build_finals_bracket(bracket, predictions)
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    leg_cols = st.columns(3)
    with leg_cols[0]:
        st.markdown(
            '<span style="display:inline-block;width:14px;height:14px;'
            'background:#e8f5e9;border:1.5px solid #00703C;border-radius:2px;'
            'vertical-align:middle;margin-right:6px;"></span>'
            '<span style="color:#1A1A1A;font-size:13px;">Locked result (winner)</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="display:inline-block;width:14px;height:14px;'
            'background:#ffebee;border:1.5px solid #ffcdd2;border-radius:2px;'
            'vertical-align:middle;margin-right:6px;"></span>'
            '<span style="color:#1A1A1A;font-size:13px;">Locked result (loser)</span>',
            unsafe_allow_html=True,
        )
    with leg_cols[1]:
        st.markdown(
            '<span style="display:inline-block;width:14px;height:14px;'
            'background:#EDE7F6;border:1.5px solid #4B2D83;border-radius:2px;'
            'vertical-align:middle;margin-right:6px;"></span>'
            '<span style="color:#1A1A1A;font-size:13px;">Model prediction (favourite)</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="display:inline-block;width:14px;height:14px;'
            'background:#FAFAFA;border:1.5px solid #E0E0E0;border-radius:2px;'
            'vertical-align:middle;margin-right:6px;"></span>'
            '<span style="color:#1A1A1A;font-size:13px;">Model prediction (underdog)</span>',
            unsafe_allow_html=True,
        )
    with leg_cols[2]:
        st.markdown(
            '<span style="display:inline-block;width:14px;height:14px;'
            'background:#F5F5F5;border:1.5px solid #E0E0E0;border-radius:2px;'
            'vertical-align:middle;margin-right:6px;"></span>'
            '<span style="color:#1A1A1A;font-size:13px;">TBD — players not yet known</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:#B8860B;font-size:13px;font-weight:700;">[N]</span>'
            f'<span style="color:#1A1A1A;font-size:13px;"> = seeded player</span>',
            unsafe_allow_html=True,
        )

with tab_q1:
    _render_quarter(bracket, predictions, quarter=1)

with tab_q2:
    _render_quarter(bracket, predictions, quarter=2)

with tab_q3:
    _render_quarter(bracket, predictions, quarter=3)

with tab_q4:
    _render_quarter(bracket, predictions, quarter=4)
