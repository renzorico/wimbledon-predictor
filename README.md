# Wimbledon 2026 Men's Singles Predictor

A statistical match prediction model for the 2026 Wimbledon Championships, built with Python, scikit-learn, XGBoost, and Streamlit. Trained on 74,848 ATP matches (2000-2026) with 41 engineered features.

## Results

| Model | Accuracy | Log Loss | ROC AUC |
|-------|----------|----------|---------|
| Weighted Elo (baseline) | 63.4% | 0.630 | 0.695 |
| Logistic Regression | 80.1% | 0.422 | 0.888 |
| **XGBoost (tuned)** | **89.7%** | **0.242** | **0.964** |

Cross-validated with `TimeSeriesSplit` to prevent temporal data leakage.

## Features

- **Surface-weighted Elo** (40% grass + 35% overall + 25% hard court)
- **Rolling serve/return stats** (last 10 matches + last 5 grass matches)
- **Head-to-head records** (all-time + grass-specific + recent 3 meetings)
- **Momentum** (win streaks, titles, form, rest days)
- **Monte Carlo simulation** (10K bracket simulations for title probabilities)

## Architecture

```
src/
├── data/           # ETL pipeline (download, load, clean)
├── features/       # Elo, rolling stats, H2H, momentum
├── models/         # WElo, LogisticRegression, XGBoost
├── simulation/     # Bracket structure + Monte Carlo engine
└── app/            # Streamlit dashboard (5 pages)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Download ATP match data
python -m src.data.download

# 2. Clean and preprocess
python -m src.data.clean

# 3. Build feature matrix (41 features × 74K matches)
python -m src.features.builder

# 4. Train and compare models
python -m src.models.compare

# 5. Launch dashboard
streamlit run src/app/Home.py
```

## Dashboard pages

| Page | Description |
|------|-------------|
| Home | Tournament status, model comparison, upsets |
| Model Comparison | Metrics, calibration curves, feature importance (SHAP) |
| Player Profiles | Radar charts, surface splits, rolling form |
| Tournament Simulation | Monte Carlo title probabilities, quarter analysis |
| Live Bracket | Seed tracker with R1 results |
| Methodology | Full technical write-up |

## Required environment variables

None. All data is fetched from public sources (TML-Database).

## Data source

[TML-Database](https://github.com/Tennismylife/TML-Database) — Sackmann-compatible ATP match database, live-updated through 2026.

## Key design decisions

- **Delta features**: all 41 features computed as Player A - Player B
- **TimeSeriesSplit**: no temporal leakage (train on past, predict future)
- **Row flipping**: half of training rows inverted to teach model symmetry
- **Surface-weighted Elo**: blends grass, overall, and hard court ratings

## Tech stack

Python, pandas, scikit-learn, XGBoost, Streamlit, Plotly, NumPy

## License

MIT
