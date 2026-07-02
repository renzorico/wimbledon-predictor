"""Methodology page — detailed explanation of the approach."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology")

st.markdown("""
## Data source

**TML-Database** — a live-updated ATP match database (Sackmann-compatible schema)
covering 1968–2026. We use matches from 2000–2026 (~75,000 completed matches) with
full serve/return statistics: aces, double faults, first-serve percentage,
break points saved, etc.

---

## Feature engineering

All features are computed as **deltas** (Player A minus Player B) so the model
learns from the relative matchup rather than absolute player strength. This
naturally encodes symmetry.

### Surface-weighted Elo (WElo)

Standard Elo treats all matches equally. But grass-court tennis is fundamentally
different from clay. Our **surface-weighted Elo** blends three separate ratings:

```
WElo = 0.40 × Elo_grass + 0.35 × Elo_overall + 0.25 × Elo_hard
```

**Why these weights?** Grass represents only ~10% of ATP matches, making a
pure grass Elo noisy. Hard court play is the closest analogue to grass
(both reward big serves), so it gets 25% weight. The overall rating provides
stability.

### Rolling serve/return statistics

Computed over a sliding window of the last 10 matches (all surfaces) and
last 5 matches (grass only):

- First serve in %, first serve won %, second serve won %
- Return points won %, break point conversion/saved rate
- Ace rate, double fault rate

### Head-to-head features

For every historical matchup between two players:
- Total H2H record (all surfaces and grass-specific)
- Recent 3 meetings winner proportion
- Default: 0.5 if players have never met

### Momentum features

- Current win streak
- Wins in last 30 days / matches in last 30 days
- Best tournament result in last 90 days
- Titles in last 12 months / grass titles in last 24 months
- Days since last match (rest vs. rust indicator)

**Total: ~41 features**, all computed chronologically to prevent data leakage.

---

## Models

### 1. Weighted Elo baseline (~63% accuracy)

No training needed. Converts the WElo delta between two players into a win
probability using the standard logistic curve:

```
P(A wins) = 1 / (1 + 10^((WElo_B - WElo_A) / 400))
```

Simple, interpretable, but ignores all features beyond Elo.

### 2. Logistic Regression (~80% accuracy)

`StandardScaler` → `LogisticRegression` using all 41 features. Provides
interpretable coefficients — we can see which features matter most
(spoiler: Elo delta and rank delta dominate).

### 3. XGBoost (~90% accuracy)

Gradient-boosted trees with hyperparameter tuning via `RandomizedSearchCV`
(50 iterations, 5-fold TimeSeriesSplit):

- `max_depth`: [3, 5, 7]
- `n_estimators`: [100, 300, 500]
- `learning_rate`: [0.01, 0.05, 0.1]
- `subsample`: [0.7, 0.8, 0.9]

Captures non-linear feature interactions (e.g., "a high-ranked player
with a recent losing streak on grass against a low-ranked grass specialist").

---

## Cross-validation strategy

We use **TimeSeriesSplit** (not random K-fold). This is critical:

- Random splits leak future information into the training set
- A player's 2025 stats would inform predictions about 2023 matches
- TimeSeriesSplit ensures we always train on past data and predict the future

**Split:** Train on 2000–2024, test on 2025–2026.

---

## Monte Carlo tournament simulation

For each of 10,000 simulations:
1. Start from the known bracket state (locked R1 results)
2. For each unplayed match, compute `P(A wins)` using the model
3. Draw winner from `Bernoulli(p)` with `numpy.random`
4. Advance winner, repeat through R2 → R3 → R4 → QF → SF → Final

**Output:** Probability of winning the title for each player, probability
of reaching each round, most likely final matchup.

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Delta features | Model predicts relative matchup, not absolute strength |
| TimeSeriesSplit | Prevents temporal data leakage |
| WElo blending | Stabilizes grass predictions for players with few grass matches |
| Row flipping during training | Half the training rows have inverted features + target=0, teaching the model symmetry |
| Parquet storage | 5-10x faster than CSV with type preservation |

---

## Limitations

1. **No in-match momentum:** We predict match outcomes, not point-by-point
2. **Injury data absent:** Player fitness is a major factor not captured
3. **Grass sample size:** Only ~10% of matches are on grass
4. **Unseeded players:** Less historical data means higher prediction uncertainty
5. **Weather and scheduling:** Not factored in

---

## References

- Bunker et al. (2024) — *Elo vs ML for Tennis Prediction* (Sage Journals)
- Ultimate Tennis Statistics — surface-specific Elo methodology
- TML-Database (GitHub) — data source
- scikit-learn, XGBoost — model implementations

---

*Built by [Renzo Rico](https://github.com/renzorico) — July 2026*
""")
