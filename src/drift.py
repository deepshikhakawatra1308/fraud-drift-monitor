"""Drift detection: PSI + KS per feature, model-score drift, and the alert rule."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples.

    Bins come from the REFERENCE quantiles, so each reference bin holds ~10%.
    PSI = sum((cur% - ref%) * ln(cur% / ref%)).
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-4, None)  # avoid log(0)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def logit(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass
class DriftThresholds:
    feature_psi: float = 0.25      # a feature counts as "drifted" above this PSI
    frac_features: float = 0.15    # retrain if >= 15% of features drifted ...
    score_psi: float = 0.25        # ... or if the model's score distribution drifted this much
    warn_score_psi: float = 0.10   # early-warning level


def check_window(ref_X: pd.DataFrame, cur_X: pd.DataFrame,
                 ref_score, cur_score, th: DriftThresholds) -> dict:
    """Compare a later window against the reference window. Needs NO labels."""
    feat_psi = {c: psi(ref_X[c].values, cur_X[c].values) for c in ref_X.columns}
    feat_ks = {c: ks_2samp(ref_X[c].values, cur_X[c].values).statistic for c in ref_X.columns}
    n_drifted = sum(v >= th.feature_psi for v in feat_psi.values())
    frac = n_drifted / len(feat_psi)
    s_psi = psi(logit(ref_score), logit(cur_score))

    if frac >= th.frac_features or s_psi >= th.score_psi:
        status = "RETRAIN"
    elif s_psi >= th.warn_score_psi or n_drifted > 0:
        status = "WARNING"
    else:
        status = "OK"
    return {"feature_psi": feat_psi, "feature_ks": feat_ks, "n_drifted": n_drifted,
            "frac_drifted": frac, "max_feature_psi": max(feat_psi.values()),
            "score_psi": s_psi, "status": status}
