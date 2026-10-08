# Fraud / Anomaly Detection with Drift Monitoring

A credit-card fraud detector that also **watches itself**: it compares a supervised model with an
unsupervised anomaly detector, picks a decision threshold from business cost, monitors incoming
data for drift (without needing labels), and raises a **RETRAIN** alert when drift crosses a bound.

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 frauds = 0.17%).

## What it does

1. Sorts transactions by time and splits **chronologically** (60% past / 40% "later") - no shuffling.
2. Trains 3 models: Logistic Regression and HistGradientBoosting (both with **class weights**) and an
**Isolation Forest** (unsupervised, never sees labels).
3. Compares them with **Precision-Recall AUC** (ROC-AUC shown only for contrast).
4. Chooses a threshold by minimising **cost** (missed fraud = lost amount, blocked legit = fixed cost).
5. Splits the later period into 4 windows and computes **PSI** (per feature + model score) per window.
6. Alert rule: `RETRAIN` if >= 15% of features have PSI >= 0.25 **or** score PSI >= 0.25.
7. Bonus: injects artificial drift into windows 3-4, shows the monitor catches it (and stays quiet on a
no-drift control), then runs a simple retrain loop.

## Setup

```bash
git clone https://github.com/<your-username>/fraud-drift-monitor.git
cd fraud-drift-monitor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle into `data/` (see `data/README.md`).

## Reproduce the results

```bash
python -m src.main                 # full run on the real data (about 1-3 minutes)
python -m src.main --synthetic     # smoke test with fake data, no download needed
```

Re-running evaluation is the same command; everything is seeded (`SEED = 42`), so numbers repeat.
Outputs go to `results/`: `metrics.json`, `monitoring\_timeline.csv`, `pr\_curves.png`,
`cost\_curve.png`, `drift\_timeline.png`, `feature\_psi.png`.

> `--synthetic` data is much easier than the real thing - never quote its numbers as results.

## Project layout

```
src/data.py    load + chronological prep + drift injection
src/drift.py   PSI, KS, alert thresholds and rule
src/costs.py   FP/FN cost and threshold search
src/main.py    the pipeline
WRITEUP.md     approach, decisions, results
```

## Tunable knobs (top of `src/main.py` and `src/drift.py`)

`FP\_COST` (cost of blocking a legit transaction), `N\_WINDOWS`, `DRIFT\_FROM`, and `DriftThresholds`.

