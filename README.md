# Fraud / Anomaly Detection with Drift Monitoring

A credit-card fraud detector that also watches itself: it compares a supervised model with an unsupervised anomaly detector, picks a decision threshold from business cost, monitors incoming data for drift (without needing labels), and raises a RETRAIN alert when drift crosses a bound.

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds = 0.17%).

## What it does

1. Sorts transactions by time and splits chronologically (60% past / 40% "later") - no shuffling.
2. Trains 3 models: Logistic Regression and HistGradientBoosting (both with class weights) and an Isolation Forest (unsupervised, never sees labels).
3. Compares them with Precision-Recall AUC (ROC-AUC shown only for contrast).
4. Chooses a threshold by minimising cost (missed fraud = lost amount, blocked legit = fixed cost).
5. Splits the later period into 4 windows and computes PSI (per feature + model score) per window.
6. Alert rule: `RETRAIN` if >= 15% of features have PSI >= 0.25 or score PSI >= 0.25.
7. Bonus: injects artificial drift into windows 3-4 and runs a simple retrain loop. Known limitation: the feature-level rule is too sensitive on this data, so the alert also fires on the no-drift control. Score PSI separates drift from no drift much better (see WRITEUP.md).

## Setup

```
git clone https://github.com/deepshikhakawatra1308/fraud-drift-monitor.git
cd fraud-drift-monitor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle into `data/` (see `data/README.md`).

## Reproduce the results

```
python -m src.main                 # full run on the real data (about 1-3 minutes)
python -m src.main --synthetic     # smoke test with fake data, no download needed
```

Everything is seeded (`SEED = 42`), so numbers repeat. Outputs go to `results/`: `metrics.json`, `monitoring_timeline.csv`, `pr_curves.png`, `cost_curve.png`, `drift_timeline.png`, `feature_psi.png`.

`--synthetic` data is much easier than the real thing - never quote its numbers as results.

## Project layout

```
src/data.py    load + chronological prep + drift injection
src/drift.py   PSI, KS, alert thresholds and rule
src/costs.py   FP/FN cost and threshold search
src/main.py    the pipeline
WRITEUP.md     approach, decisions, results
```

## Tunable knobs

Top of `src/main.py` and `src/drift.py`: `FP_COST` (cost of blocking a legit transaction), `N_WINDOWS`, `DRIFT_FROM`, and `DriftThresholds`.
