"""End-to-end pipeline: train, compare, evaluate, monitor drift, alert, retrain.

Run:  python -m src.main                      (uses data/creditcard.csv)
      python -m src.main --synthetic          (quick test with fake data)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.costs import best_threshold, total_cost
from src.data import FEATURES, inject_drift, load_real, make_synthetic
from src.drift import DriftThresholds, check_window

SEED = 42
FP_COST = 10.0        # assumed cost of blocking one legit transaction (review + friction), in currency units
N_WINDOWS = 4         # the "later" period is cut into this many consecutive windows
DRIFT_FROM = 2        # windows with index >= 2 get artificial drift (i.e. windows 3 and 4)
OUT = "results"


def pr_auc(y, s):
    return float(average_precision_score(y, s))  # average precision = standard PR-AUC estimate


def supervised(kind="hgb"):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    return HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, learning_rate=0.1,
                                          random_state=SEED)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    df = load_real(args.data) if not args.synthetic else make_synthetic()
    print(f"Rows: {len(df):,} | fraud rate: {df['Class'].mean():.4%} (frauds: {int(df['Class'].sum())})")

    # ---------- 1. chronological split (never shuffle time series!) ----------
    n = len(df)
    train_period = df.iloc[: int(0.6 * n)]
    later = df.iloc[int(0.6 * n):].reset_index(drop=True)
    fit = train_period.iloc[: int(0.75 * len(train_period))]
    val = train_period.iloc[int(0.75 * len(train_period)):]       # used for threshold + as drift reference
    print(f"fit={len(fit):,} val={len(val):,} later={len(later):,}")

    Xf, yf = fit[FEATURES], fit["Class"].values
    Xv, yv = val[FEATURES], val["Class"].values
    Xl, yl = later[FEATURES], later["Class"].values

    # ---------- 2. train three models ----------
    models = {"LogReg (class weights)": supervised("logreg"), "HistGB (class weights)": supervised("hgb")}
    for m in models.values():
        m.fit(Xf, yf)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(Xf)  # UNSUPERVISED: labels are never used

    def score(name, X):
        if name == "IsolationForest":
            return -iso.score_samples(X)         # higher = more anomalous
        return models[name].predict_proba(X)[:, 1]

    names = list(models) + ["IsolationForest"]

    # ---------- 3. compare on the clean "later" period ----------
    results = {"baseline_pr_auc_random": float(yl.mean())}
    plt.figure(figsize=(6, 5))
    for nm in names:
        s = score(nm, Xl)
        results[nm] = {"PR-AUC": pr_auc(yl, s), "ROC-AUC": float(roc_auc_score(yl, s))}
        p, r, _ = precision_recall_curve(yl, s)
        plt.plot(r, p, label=f"{nm} (PR-AUC={results[nm]['PR-AUC']:.3f})")
    plt.axhline(yl.mean(), ls="--", c="grey", label=f"random = {yl.mean():.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall, later period")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=130); plt.close()
    print("\n== Model comparison (later period, no drift) ==")
    for nm in names:
        print(f"{nm:26s} PR-AUC={results[nm]['PR-AUC']:.3f}  ROC-AUC={results[nm]['ROC-AUC']:.3f}")
    print(f"{'random guess':26s} PR-AUC={results['baseline_pr_auc_random']:.4f}")

    # ---------- 4. cost-based threshold (chosen on validation, tested on later) ----------
    champion = "HistGB (champion)"
    champ_name = "HistGB (class weights)"
    sv = score(champ_name, Xv)
    thr, table = best_threshold(yv, sv, val["Amount"].values, FP_COST)
    sl = score(champ_name, Xl)
    cost_model, fp, fn = total_cost(yl, sl, later["Amount"].values, thr, FP_COST)
    cost_none = float(later.loc[later["Class"] == 1, "Amount"].sum())   # never block anything
    cost_all = FP_COST * float((yl == 0).sum())                           # block everything
    thr_50 = 0.5
    cost_50, fp50, fn50 = total_cost(yl, sl, later["Amount"].values, thr_50, FP_COST)
    results["threshold"] = {"chosen": float(thr), "later_cost": cost_model, "later_FP": fp, "later_FN": fn,
                            "default_0.5_cost": cost_50, "default_0.5_FP": fp50, "default_0.5_FN": fn50,
                            "cost_if_no_model": cost_none, "cost_if_block_all": cost_all, "FP_COST": FP_COST}
    print(f"\n== Threshold ==\nchosen on validation: {thr:.4f}")
    print(f"later period: cost={cost_model:,.0f} (FP={fp}, FN={fn})  | threshold 0.5: cost={cost_50:,.0f} "
          f"(FP={fp50}, FN={fn50}) | no model: {cost_none:,.0f} | block all: {cost_all:,.0f}")
    plt.figure(figsize=(6, 4))
    plt.plot(table[:, 0], table[:, 1]); plt.axvline(thr, c="r", ls="--", label=f"best = {thr:.3f}")
    plt.xscale("log"); plt.xlabel("threshold (log scale)"); plt.ylabel("total cost on validation")
    plt.title("Cost vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/cost_curve.png", dpi=130); plt.close()

    # ---------- 5. drift monitoring over consecutive windows ----------
    th = DriftThresholds()
    ref_score = sv
    edges = np.linspace(0, len(later), N_WINDOWS + 1).astype(int)
    windows = [later.iloc[edges[i]: edges[i + 1]] for i in range(N_WINDOWS)]
    rows, first_drift_report = [], None
    for scenario in ("control (no drift)", "drift injected"):
        for i, w in enumerate(windows):
            w_use = inject_drift(w, train_period) if (scenario == "drift injected" and i >= DRIFT_FROM) else w
            s = score(champ_name, w_use[FEATURES])
            rep = check_window(Xv, w_use[FEATURES], ref_score, s, th)       # labels NOT used here
            has_fraud = w_use["Class"].sum() > 0
            rows.append({"scenario": scenario, "window": i + 1,
                         "max_feature_PSI": rep["max_feature_psi"],
                         "features_drifted": f"{rep['n_drifted']}/{len(FEATURES)}",
                         "score_PSI": rep["score_psi"],
                         "flag_rate_%": 100 * float((s >= thr).mean()),
                         "PR-AUC (needs labels)": pr_auc(w_use["Class"].values, s) if has_fraud else np.nan,
                         "status": rep["status"]})
            if scenario == "drift injected" and i == DRIFT_FROM:
                first_drift_report = rep
    mon = pd.DataFrame(rows)
    mon.to_csv(f"{OUT}/monitoring_timeline.csv", index=False)
    print("\n== Drift monitoring (reference = validation window) ==")
    print(mon.round(3).to_string(index=False))

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sc, mk in (("control (no drift)", "o-"), ("drift injected", "s-")):
        d = mon[mon.scenario == sc]
        ax[0].plot(d.window, d.score_PSI, mk, label=sc)
        ax[1].plot(d.window, d["PR-AUC (needs labels)"], mk, label=sc)
    ax[0].axhline(th.score_psi, c="r", ls="--", label="retrain bound"); ax[0].axhline(th.warn_score_psi, c="orange", ls=":", label="warning")
    ax[0].set_title("Score drift (PSI) per window"); ax[0].set_xlabel("window"); ax[0].legend(fontsize=8)
    ax[1].set_title("Model PR-AUC per window (delayed labels)"); ax[1].set_xlabel("window"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/drift_timeline.png", dpi=130); plt.close()

    top = sorted(first_drift_report["feature_psi"].items(), key=lambda kv: -kv[1])[:10]
    plt.figure(figsize=(6, 4))
    plt.barh([k for k, _ in top][::-1], [v for _, v in top][::-1])
    plt.axvline(th.feature_psi, c="r", ls="--"); plt.xlabel("PSI"); plt.title("Top drifted features (window 3, injected)")
    plt.tight_layout(); plt.savefig(f"{OUT}/feature_psi.png", dpi=130); plt.close()

    # ---------- 6. alert -> retrain loop ----------
    d = mon[mon.scenario == "drift injected"]
    alert_windows = d.loc[d.status == "RETRAIN", "window"].tolist()
    print(f"\nWindows that raised RETRAIN: {alert_windows}")
    if alert_windows:
        k = alert_windows[0] - 1                       # 0-based index of the first alerted window
        if k + 1 < N_WINDOWS:
            w_alert = inject_drift(windows[k], train_period)
            w_next = inject_drift(windows[k + 1], train_period)
            stale_s = score(champ_name, w_next[FEATURES])
            # labels for the alerted window have now arrived -> retrain on old data + that window
            # (the recent window is up-weighted 5x so the model pays attention to the new behaviour)
            train_new = pd.concat([fit, val, w_alert])
            sw = np.where(np.arange(len(train_new)) >= len(fit) + len(val), 5.0, 1.0)
            new_model = supervised("hgb")
            new_model.fit(train_new[FEATURES], train_new["Class"], sample_weight=sw)
            new_s = new_model.predict_proba(w_next[FEATURES])[:, 1]
            yn = w_next["Class"].values
            results["retrain_loop"] = {"alert_window": k + 1, "evaluated_on_window": k + 2,
                                       "stale_model_PR-AUC": pr_auc(yn, stale_s),
                                       "retrained_model_PR-AUC": pr_auc(yn, new_s)}
            print(f"Retrain loop: stale PR-AUC={pr_auc(yn, stale_s):.3f} -> retrained PR-AUC={pr_auc(yn, new_s):.3f} "
                  f"(evaluated on window {k + 2})")
    results["monitoring_alert_windows_drift_scenario"] = alert_windows
    results["monitoring_alert_windows_control"] = mon.loc[(mon.scenario == "control (no drift)") &
                                                          (mon.status == "RETRAIN"), "window"].tolist()
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved plots + metrics.json + monitoring_timeline.csv to ./{OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--synthetic", action="store_true", help="use fake data (for quick testing only)")
    main(ap.parse_args())
