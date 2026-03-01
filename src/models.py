"""
ML model definitions and stability selection.

Five nested models:
  A  Simoes-3          — fixed 3 features: APLP1, CHL1, MAPT
  B  Random-3          — 3 random non-endosomal proteins (negative control)
  C  Endosomal-Full    — all endosomal proteins, elastic net
  D  Endosomal-Optimised — stability-selected subset, L2 logistic
  E  Endosomal-XGB     — XGBoost on stability-selected features

IMPORTANT VERSION NOTES:
  - XGBoost ≥ 2.0: no use_label_encoder; eval_metric in constructor only
  - sklearn ≥ 1.7: no multi_class param in LogisticRegression
  - NumPy: use np.random.default_rng(), not legacy RandomState
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import xgboost as xgb

from src.config import (
    RANDOM_STATE,
    ELASTICNET_L1_RATIO, ELASTICNET_C, ELASTICNET_MAX_ITER,
    XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_N_ESTIMATORS,
    STABILITY_N_ITERATIONS, STABILITY_N_ITERATIONS_CV,
    STABILITY_SAMPLE_FRACTION, STABILITY_SAMPLE_FRACTION_SMALL,
    STABILITY_THRESHOLD,
)

# Known endosomal proteins (mouse gene symbols, lowercase for matching)
ENDOSOMAL_PROTEINS = {
    "aplp1", "aplp2", "app", "chl1", "mapt", "sorl1", "sort1",
    "ctsd", "ctsb", "lamp1", "lamp2", "cadm1", "cadm4", "nrxn1",
    "nrcam", "l1cam", "ncam1", "thy1", "cntn1", "nfasc", "negr1",
    "opcml", "ntm", "sez6l", "sez6", "clstn1", "clstn3",
    "bsg", "cd47", "itm2b",
}

SIMOES_3 = ["Aplp1", "Chl1", "Mapt"]   # canonical capitalisation


def get_endosomal_features(all_feature_names: list[str]) -> list[str]:
    """Return feature names matching endosomal protein set (case-insensitive)."""
    return [f for f in all_feature_names if f.lower() in ENDOSOMAL_PROTEINS]


def get_random_3_features(
    all_feature_names: list[str],
    exclude: set[str] | None = None,
    seed: int = RANDOM_STATE,
) -> list[str]:
    """Pick 3 random non-endosomal features for the negative control."""
    rng = np.random.default_rng(seed)
    pool = [
        f for f in all_feature_names
        if f.lower() not in ENDOSOMAL_PROTEINS
        and (exclude is None or f not in exclude)
        and f != "label"
    ]
    return list(rng.choice(pool, size=min(3, len(pool)), replace=False))


# ── Model definitions ─────────────────────────────────────────────────────────

def get_model_definitions(class_weight_ratio: float = 1.0) -> dict[str, Pipeline]:
    """
    Return a dict of {model_name: sklearn Pipeline}.

    class_weight_ratio: pos_weight for XGBoost (n_negative / n_positive).
    """
    models: dict[str, Pipeline] = {}

    # ── Model A: Simoes-3 ──
    models["Simoes-3"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=5000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # ── Model B: Random-3 ──
    models["Random-3"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=5000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # ── Model C: Endosomal-Full ──
    models["Endosomal-Full"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=ELASTICNET_L1_RATIO, C=ELASTICNET_C,
            max_iter=ELASTICNET_MAX_ITER, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # ── Model D: Endosomal-Optimised ──
    models["Endosomal-Optimized"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=5000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # ── Model E: Endosomal-XGB ──
    # XGBoost ≥ 2.0: no use_label_encoder; eval_metric in constructor only
    models["Endosomal-XGB"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=5.0,
            scale_pos_weight=class_weight_ratio,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])

    return models


# ── Stability selection ───────────────────────────────────────────────────────

def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = STABILITY_N_ITERATIONS,
    sample_fraction: float = STABILITY_SAMPLE_FRACTION,
    C_range: np.ndarray | None = None,
    threshold: float = STABILITY_THRESHOLD,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Bootstrapped L1 logistic regression stability selection.

    For each iteration:
      1. Subsample sample_fraction of data WITHOUT replacement
      2. Fit L1 logistic regression at a random C
      3. Record which features have non-zero coefficients

    Parameters
    ----------
    X             : (n_samples, n_features) array, already standardised
    y             : (n_samples,) binary labels
    n_iterations  : number of bootstrap iterations (≥500 recommended)
    sample_fraction: fraction of samples per iteration
    C_range       : regularisation values to sample from
    threshold     : minimum selection probability to call a feature stable
    seed          : RNG seed

    Returns
    -------
    selection_probs : np.ndarray of shape (n_features,)
    """
    if C_range is None:
        C_range = np.logspace(-3, 1, 20)

    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    selection_counts = np.zeros(n_features)
    sub_size = max(2, int(n_samples * sample_fraction))

    for i in range(n_iterations):
        idx = rng.choice(n_samples, size=sub_size, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]

        # Skip iteration if only one class present
        if len(np.unique(y_sub)) < 2:
            continue

        C = float(rng.choice(C_range))
        try:
            model = LogisticRegression(
                penalty="l1", C=C, solver="saga",
                max_iter=5000,
                random_state=int(rng.integers(1_000_000)),
                class_weight="balanced",
            )
            model.fit(X_sub, y_sub)
            if model.coef_ is not None and not np.any(np.isnan(model.coef_)):
                selected = np.abs(model.coef_[0]) > 1e-8
                selection_counts += selected
        except Exception:
            continue

    return selection_counts / n_iterations


def get_stable_features(
    selection_probs: np.ndarray,
    feature_names: list[str],
    threshold: float = STABILITY_THRESHOLD,
    min_features: int = 2,
) -> list[str]:
    """
    Return feature names whose selection probability ≥ threshold.
    Falls back to top-k by probability if fewer than min_features qualify.
    """
    stable = [name for name, prob in zip(feature_names, selection_probs)
              if prob >= threshold]
    if len(stable) < min_features:
        top_idx = np.argsort(selection_probs)[-min_features:][::-1]
        stable  = [feature_names[i] for i in top_idx]
    return stable


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_extraction import load_all_data
    from src.preprocessing import preprocess_expression_matrix, prepare_ml_arrays

    data = load_all_data()
    matrix = preprocess_expression_matrix(data["mouse_matrix"])
    X, y, names = prepare_ml_arrays(matrix)

    print(f"[models] Feature matrix: {X.shape}, labels: {np.bincount(y)}")

    endo = get_endosomal_features(names)
    rand = get_random_3_features(names)
    print(f"  Endosomal features: {len(endo)}")
    print(f"  Random-3 features:  {rand}")

    models = get_model_definitions()
    print(f"  Model definitions:  {list(models.keys())}")

    # Quick stability selection (few iterations for smoke-test)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X[:, [names.index(f) for f in endo if f in names]])
    probs  = stability_selection(X_sc, y, n_iterations=50, sample_fraction=0.5)
    stable = get_stable_features(probs, endo)
    print(f"  Stable features (50 iter): {stable}")
