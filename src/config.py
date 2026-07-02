"""Project configuration — paths, constants, hyperparameters."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WIMBLEDON_2026_DIR = DATA_DIR / "wimbledon_2026"
MODELS_DIR = ROOT_DIR / "models"

# ── Data source ───────────────────────────────────────────────────────────
# TML-Database: Sackmann-compatible schema, live-updated through 2026
MATCH_YEARS = range(2000, 2027)  # 2000–2026 inclusive

# ── Elo configuration ────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K_FACTOR = 32
# Surface-weighted Elo blend for grass prediction
WELO_WEIGHTS = {"grass": 0.40, "overall": 0.35, "hard": 0.25}

# ── Feature engineering ───────────────────────────────────────────────────
ROLLING_WINDOW = 10          # Last N matches for rolling stats
ROLLING_WINDOW_GRASS = 5     # Last N grass matches
MOMENTUM_DAYS_SHORT = 30     # Recent form window
MOMENTUM_DAYS_LONG = 90      # Best-result window
H2H_RECENT_N = 3             # Recent H2H meetings

# ── Model training ────────────────────────────────────────────────────────
CV_SPLITS = 5
RANDOM_STATE = 42
TEST_YEAR_CUTOFF = 2025      # 2025+ held out for testing
VAL_YEAR_CUTOFF = 2023       # 2023-2024 for validation

XGB_SEARCH_SPACE = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}
XGB_SEARCH_ITERS = 50
XGB_SEARCH_JOBS = 1

# ── Simulation ────────────────────────────────────────────────────────────
MONTE_CARLO_SIMS = 10_000
WIMBLEDON_2026_START_DATE = "2026-06-29"

# ── Round encoding (for features) ────────────────────────────────────────
ROUND_DEPTH = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    # Sackmann aliases
    "RR": 1,
}

# ── Tournament levels ────────────────────────────────────────────────────
TOURNEY_LEVEL_MAP = {
    "G": 4,   # Grand Slam
    "M": 3,   # Masters 1000
    "A": 2,   # ATP 500/250
    "D": 1,   # Davis Cup
    "F": 3,   # Tour Finals
}

# ── Surfaces ──────────────────────────────────────────────────────────────
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
