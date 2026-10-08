"""Data loading, chronological splitting, and artificial drift injection."""
import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]

# Features that separate fraud from legit best in the Kaggle data.
# Used only to simulate "fraudsters change tactics" (concept drift).
KEY_FRAUD_FEATURES = ["V14", "V12", "V10", "V17", "V3", "V4", "V11"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time (so 'later' really means later) and add log(amount)."""
    df = df.sort_values("Time").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["Amount"])
    return df


def load_real(path: str) -> pd.DataFrame:
    return prepare(pd.read_csv(path))


def make_synthetic(n: int = 80_000, fraud_rate: float = 0.004, seed: int = 42) -> pd.DataFrame:
    """Fake data shaped like the Kaggle file. ONLY for quick testing without the CSV."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    shift = {2: -2.0, 3: 2.0, 9: -2.0, 10: 1.5, 11: -2.2, 13: -2.5, 16: -2.0}  # col index -> mean shift for fraud
    for j, s in shift.items():
        X[y == 1, j] += s + rng.normal(0, 0.3, size=(y == 1).sum())
    amount = np.where(y == 1, rng.lognormal(4.0, 1.5, n), rng.lognormal(3.3, 1.2, n))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.uniform(0, 172_800, n))
    df["Amount"] = amount
    df["Class"] = y
    return prepare(df)


def inject_drift(window: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `window` with artificial drift added.

    1) Covariate drift: shift a few features for ALL transactions
       (e.g. new payment habits, a new merchant mix, inflation in amounts).
    2) Concept drift: fraud rows get pulled toward the legit average on the
       key features (fraudsters learn to look normal), so an old model misses them.
    `ref` (the training-period data) supplies means/stds for the shifts.
    """
    out = window.copy()
    std = ref[FEATURES].std()
    legit_mean = ref.loc[ref["Class"] == 0, FEATURES].mean()

    # 1) covariate drift, sized in "training standard deviations"
    for col, k in {"V4": 1.5, "V11": 1.0, "V17": -1.0, "log_amount": 0.8}.items():
        out[col] = out[col] + k * std[col]

    # 2) concept drift: fraud mimics legit behaviour
    is_fraud = out["Class"] == 1
    alpha = 0.6  # 0 = no change, 1 = fraud looks exactly like average legit
    for col in KEY_FRAUD_FEATURES:
        out.loc[is_fraud, col] = (1 - alpha) * out.loc[is_fraud, col] + alpha * legit_mean[col]
    return out
