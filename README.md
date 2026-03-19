# March Madness 2026

A machine learning pipeline that predicts win probabilities for every possible matchup in the 2026 NCAA Tournament (Men's and Women's), producing a Kaggle-compatible `submission.csv` for the [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

---

## How It Works

The model follows a five-step pipeline, run independently for each gender:

1. **ELO Ratings** — Compute per-team ELO ratings across all historical regular seasons and tournaments to capture long-run team strength.
2. **Feature Engineering** — For every past tournament game, build a rich feature vector that includes:
   - Regular-season stats (points, rebounds, FG%, etc.) for both teams
   - ELO delta at game time
   - Tournament seed difference
   - Massey ranking ordinals (Men's only)
   - Prior tournament win percentage
3. **Walk-forward Training** — Train a `RandomForestClassifier` using an expanding window: train on all seasons up to year *N−1*, evaluate on year *N*, then advance. This avoids look-ahead bias and realistically simulates how the model would perform live.
4. **2026 Predictions** — Apply the trained classifier to generate a win probability for every possible 2026 tournament matchup pair.
5. **Submission** — Non-tournament matchups are assigned a neutral 0.5 probability; all pairs are written to `kaggle/working/submission.csv`.

---

## Model Details

| | Men's | Women's |
|---|---|---|
| Algorithm | Random Forest | Random Forest |
| Training data from | 2003 | 2009 |
| Massey rankings | ✅ | ❌ (unavailable) |
| Features | ELO, seeds, reg-season stats, Massey, prior tourney % | ELO, seeds, reg-season stats, prior tourney % |

---

## 2026 Men's Bracket Predictions

Picks derived by simulating each round using `simulate_bracket.py` — the team with the higher predicted win probability advances.

### First Four
| Game | Predicted Winner |
|---|---|
| Lehigh vs Prairie View | Prairie View |
| Howard vs UMBC | UMBC |
| Miami OH vs SMU | SMU |
| NC State vs Texas | NC State |

### Region Winners
| Region | Winner |
|---|---|
| West (W) | Connecticut |
| East (X) | Florida |
| South (Y) | Michigan |
| Midwest (Z) | Arizona |

### Final Four
| Semifinal | Winner |
|---|---|
| Connecticut vs Florida | Connecticut |
| Michigan vs Arizona | Michigan |

### 🏆 Champion: **Michigan**

---

## Project Structure

```
predictor.ipynb        # Main ML pipeline (feature engineering, training, prediction)
simulate_bracket.py    # Simulates the full bracket round-by-round using model predictions
kaggle/
  input/
    competitions/
      march-machine-learning-mania-2026/   # Raw Kaggle data files
  working/
    submission.csv       # Output: win probability for every possible matchup pair
```

## Kaggle Notebook

The notebook is also published on Kaggle: [March Madness 2026](https://www.kaggle.com/code/caiomelo22/march-madness-2026)

> **Note:** This link will not be publicly accessible until the competition ends.

---

## Running Locally

1. Install dependencies: `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
2. Open and run all cells in `predictor.ipynb` — this populates `final_df` and writes `submission.csv`
3. To simulate the bracket: `python simulate_bracket.py`

