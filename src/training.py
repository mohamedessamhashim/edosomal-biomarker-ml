"""
Training loop: repeated stratified k-fold cross-validation across all 5 models.

Key design decisions:
  - Same CV splits used for all models (fair comparison)
  - Stability selection for Model D runs INSIDE each CV fold (no data leakage)
  - StandardScaler fit on training fold, applied to validation fold
  - Paired t-test for model comparison (same folds → paired)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from scipy import stats

from src.config import (
    RANDOM_STATE, CV_N_SPLITS, CV_N_REPEATS,
    STABILITY_N_ITERATIONS_CV, STABILITY_SAMPLE_FRACTION_SMALL,
    STABILITY_THRESHOLD, RESULTS_DIR,
)
from src.models import (
    get_model_definitions, get_endosomal_features, get_random_3_features,
    stability_selection, get_stable_features, SIMOES_3,
)


# ── Feature set builder ───────────────────────────────────────────────────────

def build_feature_sets(
    feature_names: list[str],
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
) -> dict[str, list[str]]:
    """
    Determine which features each model uses.

    Model A (Simoes-3)       : fixed list [Aplp1, Chl1, Mapt]
    Model B (Random-3)       : 3 random non-endosomal proteins
    Model C (Endosomal-Full) : all endosomal proteins found in feature_names
    Model D (Endosomal-Opt.) : determined inside CV folds (returned as 'all_endosomal')
    Model E (Endosomal-XGB)  : same as Model D — stability-selected inside CV
    """
    endosomal = get_endosomal_features(feature_names)
    simoes_avail = [f for f in SIMOES_3 if f in feature_names]

    # If the exact Simoes-3 names are not present, do case-insensitive match
    if len(simoes_avail) < 3:
        lower_map = {f.lower(): f for f in feature_names}
        simoes_avail = [lower_map[s.lower()] for s in SIMOES_3 if s.lower() in lower_map]

    return {
        "Simoes-3":           simoes_avail,
        "Random-3":           get_random_3_features(feature_names),
        "Endosomal-Full":     endosomal,
        "Endosomal-Optimized": endosomal,   # refined inside each fold
        "Endosomal-XGB":      endosomal,    # refined inside each fold
    }


# ── Main CV loop ──────────────────────────────────────────────────────────────

def run_model_comparison(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv_config: dict | None = None,
    n_stability_iter: int = STABILITY_N_ITERATIONS_CV,
    stability_fraction: float = STABILITY_SAMPLE_FRACTION_SMALL,
    verbose: bool = True,
) -> dict:
    """
    Run all 5 models with repeated stratified k-fold CV.

    Parameters
    ----------
    X             : (n_samples, n_features) float array
    y             : (n_samples,) binary int array
    feature_names : list of column names corresponding to X columns
    cv_config     : dict with keys n_splits, n_repeats, seed
    n_stability_iter : stability selection iterations per fold (Model D/E)
    stability_fraction : subsample fraction for stability selection

    Returns
    -------
    results : dict[model_name → metrics_dict]
    """
    if cv_config is None:
        cv_config = {"n_splits": CV_N_SPLITS, "n_repeats": CV_N_REPEATS, "seed": RANDOM_STATE}

    cv = RepeatedStratifiedKFold(
        n_splits=cv_config["n_splits"],
        n_repeats=cv_config["n_repeats"],
        random_state=cv_config["seed"],
    )

    models_def = get_model_definitions(
        class_weight_ratio=float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1)
    )
    feature_sets = build_feature_sets(feature_names, X, y)

    results: dict[str, dict] = {name: {
        "aucs": [], "all_y_true": [], "all_y_prob": [], "n_features_per_fold": []
    } for name in models_def}

    splits = list(cv.split(X, y))

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if verbose and fold_i % cv_config["n_splits"] == 0:
            print(f"  Repeat {fold_i // cv_config['n_splits'] + 1}/{cv_config['n_repeats']}")

        for model_name, base_pipeline in models_def.items():
            feat_cols = feature_sets[model_name]
            feat_idx  = [feature_names.index(f) for f in feat_cols if f in feature_names]
            if not feat_idx:
                continue

            X_sub = X[:, feat_idx]
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Stability selection inside the fold (Models D and E only)
            if model_name in ("Endosomal-Optimized", "Endosomal-XGB"):
                scaler_inner = StandardScaler()
                X_tr_sc = scaler_inner.fit_transform(X_tr)
                sel_probs = stability_selection(
                    X_tr_sc, y_tr,
                    n_iterations=n_stability_iter,
                    sample_fraction=stability_fraction,
                    seed=RANDOM_STATE + fold_i,
                )
                stable_mask = sel_probs >= STABILITY_THRESHOLD
                if stable_mask.sum() < 2:
                    top3 = np.argsort(sel_probs)[-3:]
                    stable_mask = np.zeros(len(sel_probs), dtype=bool)
                    stable_mask[top3] = True

                X_tr  = X_tr[:, stable_mask]
                X_val = X_val[:, stable_mask]
                results[model_name]["n_features_per_fold"].append(int(stable_mask.sum()))
            else:
                results[model_name]["n_features_per_fold"].append(len(feat_idx))

            # Guard: single-class folds
            if len(np.unique(y_val)) < 2:
                if verbose:
                    print(f"    Warning: {model_name} fold {fold_i} single-class — skipped")
                continue

            fold_pipeline = clone(base_pipeline)
            try:
                fold_pipeline.fit(X_tr, y_tr)
                y_prob = fold_pipeline.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob)
                results[model_name]["aucs"].append(auc)
                results[model_name]["all_y_true"].extend(y_val.tolist())
                results[model_name]["all_y_prob"].extend(y_prob.tolist())
            except Exception as e:
                if verbose:
                    print(f"    Warning: {model_name} fold {fold_i} failed: {e}")
                continue

    # Compute summary statistics
    for name, res in results.items():
        aucs = res["aucs"]
        y_true_all = np.array(res["all_y_true"])
        y_prob_all = np.array(res["all_y_prob"])
        n_feat = res["n_features_per_fold"]

        res["mean_auc"]    = float(np.mean(aucs)) if aucs else 0.0
        res["std_auc"]     = float(np.std(aucs))  if aucs else 0.0
        res["ci_95_low"]   = float(np.percentile(aucs, 2.5))  if aucs else 0.0
        res["ci_95_high"]  = float(np.percentile(aucs, 97.5)) if aucs else 0.0
        res["n_folds"]     = len(aucs)
        res["n_features"]  = int(np.median(n_feat)) if n_feat else len(feature_sets.get(name, []))
        res["all_y_true"]  = y_true_all
        res["all_y_prob"]  = y_prob_all

        if verbose:
            print(
                f"  {name:25s}: AUC = {res['mean_auc']:.3f} ± {res['std_auc']:.3f} "
                f"(95% CI: {res['ci_95_low']:.3f}–{res['ci_95_high']:.3f}), "
                f"n_features = {res['n_features']}"
            )

    return results


# ── Statistical model comparison ──────────────────────────────────────────────

def compare_models_paired(
    results: dict,
    model_a: str,
    model_b: str,
) -> dict:
    """
    Paired t-test on CV fold AUCs.

    Both models must have been evaluated on the same CV splits.
    """
    aucs_a = results[model_a]["aucs"]
    aucs_b = results[model_b]["aucs"]
    min_len = min(len(aucs_a), len(aucs_b))
    aucs_a, aucs_b = aucs_a[:min_len], aucs_b[:min_len]
    t_stat, p_value = stats.ttest_rel(aucs_a, aucs_b)
    delta = np.mean(aucs_b) - np.mean(aucs_a)
    return {
        "model_a":   model_a,
        "model_b":   model_b,
        "delta_auc": float(delta),
        "t_stat":    float(t_stat),
        "p_value":   float(p_value),
    }


# ── Per-threshold metrics ─────────────────────────────────────────────────────

def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity":  specificity,
        "ppv":         tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "npv":         tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_extraction import load_all_data
    from src.preprocessing import preprocess_expression_matrix, prepare_ml_arrays

    data = load_all_data()
    matrix = preprocess_expression_matrix(data["mouse_matrix"])
    X, y, names = prepare_ml_arrays(matrix)

    print(f"[training] Running 5-fold × 3-repeat CV on {X.shape[0]} samples, "
          f"{X.shape[1]} features …")

    results = run_model_comparison(
        X, y, names,
        cv_config={"n_splits": 5, "n_repeats": 3, "seed": RANDOM_STATE},
        n_stability_iter=50,
    )

    print("\nComparisons vs Simoes-3:")
    for name in ["Endosomal-Full", "Endosomal-Optimized", "Endosomal-XGB"]:
        if name in results:
            cmp = compare_models_paired(results, "Simoes-3", name)
            print(f"  {name}: ΔAUC = {cmp['delta_auc']:+.3f}, p = {cmp['p_value']:.4f}")

    # Save
    from src.utils import generate_comparison_table
    table = generate_comparison_table(results)
    out = RESULTS_DIR / "model_comparison_table.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved comparison table to {out}")
