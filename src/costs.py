"""Business-cost analysis: false positives vs false negatives."""
import numpy as np


def total_cost(y, score, amount, threshold, fp_cost, fn_fixed=0.0):
    """Cost of using `threshold`.
    FN (missed fraud)   -> we lose the transaction amount (+ optional fixed chargeback fee)
    FP (blocked legit)  -> fixed cost for review / customer friction / lost sale
    """
    flagged = score >= threshold
    fn = (~flagged) & (y == 1)
    fp = flagged & (y == 0)
    return float(amount[fn].sum() + fn_fixed * fn.sum() + fp_cost * fp.sum()), int(fp.sum()), int(fn.sum())


def best_threshold(y, score, amount, fp_cost, fn_fixed=0.0, n_candidates=400):
    """Scan thresholds, return (best_threshold, table of [threshold, cost, fp, fn])."""
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, n_candidates)))
    rows = [(t, *total_cost(y, score, amount, t, fp_cost, fn_fixed)) for t in cands]
    best = min(rows, key=lambda r: r[1])
    return best[0], np.array(rows)
