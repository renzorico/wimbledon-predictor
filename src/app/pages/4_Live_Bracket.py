"""Live bracket page — visual draw with results and predictions."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Live Bracket", layout="wide")
st.title("Live bracket — R1 results")

# ── Seed data with results ────────────────────────────────────────────────
SEED_RESULTS = [
    {"Seed": 1, "Player": "Jannik Sinner", "Q": "Q1", "R1": "W", "Opp": "M. Kecmanovic", "Score": "4-6 6-3 6-7(8) 6-2 6-3"},
    {"Seed": 2, "Player": "Alexander Zverev", "Q": "Q4", "R1": "W", "Opp": "A. Blockx", "Score": "6-4 6-7(8) 7-6(5) 7-6(0)"},
    {"Seed": 3, "Player": "Felix Auger-Aliassime", "Q": "Q2", "R1": "W", "Opp": "A. Shevchenko", "Score": "6-3 6-1 6-4"},
    {"Seed": 4, "Player": "Ben Shelton", "Q": "Q3", "R1": "L", "Opp": "O. Virtanen", "Score": "6-4 3-6 6-7(8) 2-6 6-7(9)"},
    {"Seed": 5, "Player": "Alex de Minaur", "Q": "Q1", "R1": "W", "Opp": "R. Burruchaga", "Score": "7-6(5) 6-1 6-0"},
    {"Seed": 6, "Player": "Taylor Fritz", "Q": "Q4", "R1": "W", "Opp": "D. Lajovic", "Score": "6-3 6-4 6-3"},
    {"Seed": 7, "Player": "Novak Djokovic", "Q": "Q2", "R1": "W", "Opp": "Y. Wu", "Score": "6-4 5-7 6-4 6-4"},
    {"Seed": 8, "Player": "Daniil Medvedev", "Q": "Q3", "R1": "W", "Opp": "M. Cilic", "Score": "6-1 6-2 6-4"},
    {"Seed": 9, "Player": "Flavio Cobolli", "Q": "Q1", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 10, "Player": "Alexander Bublik", "Q": "Q4", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 11, "Player": "Casper Ruud", "Q": "Q2", "R1": "L", "Opp": "H. Hurkacz", "Score": "4-6 2-6 6-7(7)"},
    {"Seed": 12, "Player": "Andrey Rublev", "Q": "Q3", "R1": "L", "Opp": "R. Safiullin", "Score": "4-6 7-6(6) 3-6 6-3 6-7(12)"},
    {"Seed": 13, "Player": "Jiri Lehecka", "Q": "Q1", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 14, "Player": "Luciano Darderi", "Q": "Q4", "R1": "L", "Opp": "E. Quinn", "Score": "6-7(7) 5-7 2-6"},
    {"Seed": 15, "Player": "Jakub Mensik", "Q": "Q2", "R1": "W", "Opp": "T. Samuel", "Score": "7-5 3-6 3-6 6-3 7-6(7)"},
    {"Seed": 16, "Player": "Learner Tien", "Q": "Q3", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 17, "Player": "Frances Tiafoe", "Q": "Q1", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 18, "Player": "Francisco Cerundolo", "Q": "Q4", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 19, "Player": "Karen Khachanov", "Q": "Q2", "R1": "W", "Opp": "B. Harris", "Score": "6-3 7-5 6-3 6-3"},
    {"Seed": 20, "Player": "Arthur Fils", "Q": "Q3", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 21, "Player": "Tommy Paul", "Q": "Q1", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 22, "Player": "A. Davidovich Fokina", "Q": "Q4", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 23, "Player": "Rafael Jodar", "Q": "Q2", "R1": "W", "Opp": "F. Gill", "Score": "6-3 6-3 7-5"},
    {"Seed": 24, "Player": "Joao Fonseca", "Q": "Q3", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 25, "Player": "Arthur Rinderknech", "Q": "Q1", "R1": "W", "Opp": "O. Tarvet", "Score": "7-6(4) 7-6(4) 4-6 7-5"},
    {"Seed": 26, "Player": "Cameron Norrie", "Q": "Q4", "R1": "L", "Opp": "M. Zheng", "Score": "7-6(7) 2-6 7-6(2) 3-6 6-7(4)"},
    {"Seed": 27, "Player": "Ugo Humbert", "Q": "Q2", "R1": "L", "Opp": "Z. Bergs", "Score": "2-6 5-7 6-4 6-3 3-6"},
    {"Seed": 28, "Player": "Brandon Nakashima", "Q": "Q3", "R1": "W", "Opp": "J.P. Jones", "Score": "6-3 7-6(5) ..."},
    {"Seed": 29, "Player": "T.M. Etcheverry", "Q": "Q1", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 30, "Player": "Alejandro Tabilo", "Q": "Q4", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 31, "Player": "Ignacio Buse", "Q": "Q2", "R1": "—", "Opp": "TBD", "Score": ""},
    {"Seed": 32, "Player": "Matteo Arnaldi", "Q": "Q3", "R1": "—", "Opp": "TBD", "Score": ""},
]

df = pd.DataFrame(SEED_RESULTS)

# ── Quarter filter ────────────────────────────────────────────────────────
quarter = st.selectbox("Quarter", ["All", "Q1", "Q2", "Q3", "Q4"])
if quarter != "All":
    df = df[df["Q"] == quarter]

# ── Summary metrics ───────────────────────────────────────────────────────
total_seeds = len(SEED_RESULTS)
seeds_out = sum(1 for s in SEED_RESULTS if s["R1"] == "L")
seeds_through = sum(1 for s in SEED_RESULTS if s["R1"] == "W")
seeds_pending = total_seeds - seeds_out - seeds_through

c1, c2, c3 = st.columns(3)
c1.metric("Seeds through R1", seeds_through)
c2.metric("Seeds eliminated", seeds_out, delta=f"-{seeds_out}", delta_color="inverse")
c3.metric("Pending", seeds_pending)

# ── Color-coded bracket table ─────────────────────────────────────────────
st.subheader("Seed tracker")

def color_result(val):
    if val == "W":
        return "background-color: #0a3d1a; color: #4ade80;"
    elif val == "L":
        return "background-color: #3d0a0a; color: #f87171;"
    return ""

styled = df.style.map(color_result, subset=["R1"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=700)

# ── Five-set thrillers ────────────────────────────────────────────────────
st.subheader("Five-set matches")
five_setters = [s for s in SEED_RESULTS if s["Score"].count("-") >= 5]
if five_setters:
    st.dataframe(pd.DataFrame(five_setters), use_container_width=True, hide_index=True)
else:
    st.info("No five-set matches tracked yet.")
