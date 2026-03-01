"""
SHAP interpretability for all model types.

CRITICAL: When a model is a Pipeline, extract the fitted classifier and
pass already-scaled data to the SHAP explainer — otherwise SHAP values
are computed on the wrong scale.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR, RANDOM_STATE


def _extract_clf_and_scale(pipeline: Pipeline, X_train: np.ndarray, X_explain: np.ndarray):
    """
    Extract the fitted classifier and return scaled arrays.

    Returns
    -------
    clf        : fitted classifier (not the full pipeline)
    X_tr_sc    : scaled training data
    X_ex_sc    : scaled explanation data
    """
    if hasattr(pipeline, "named_steps"):
        steps = list(pipeline.named_steps.items())
        # Apply all transformers except the final estimator
        X_tr = X_train.copy()
        X_ex = X_explain.copy()
        for name, step in steps[:-1]:
            X_tr = step.transform(X_tr)
            X_ex = step.transform(X_ex)
        clf = steps[-1][1]
    else:
        clf  = pipeline
        X_tr = X_train
        X_ex = X_explain

    return clf, X_tr, X_ex


def compute_shap_tree(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    """
    SHAP TreeExplainer for XGBoost (or other tree models).

    Returns
    -------
    shap_values    : (n_explain, n_features) array
    expected_value : float baseline
    """
    import shap

    clf, X_tr_sc, X_ex_sc = _extract_clf_and_scale(pipeline, X_train, X_explain)

    explainer = shap.TreeExplainer(
        clf,
        data=X_tr_sc,
        feature_perturbation="interventional",
    )
    shap_values = explainer.shap_values(X_ex_sc)

    # Older XGBoost returns a list for binary classification — take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, float(explainer.expected_value)


def compute_shap_linear(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    """
    SHAP LinearExplainer for logistic regression models.

    Returns
    -------
    shap_values    : (n_explain, n_features) array
    expected_value : float baseline
    """
    import shap

    clf, X_tr_sc, X_ex_sc = _extract_clf_and_scale(pipeline, X_train, X_explain)

    explainer = shap.LinearExplainer(
        clf,
        X_tr_sc,
        feature_perturbation="correlation_dependent",
    )
    shap_values = explainer.shap_values(X_ex_sc)
    return shap_values, float(explainer.expected_value)


def compute_shap_values(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    model_type: str = "auto",
) -> tuple[np.ndarray, float]:
    """
    Dispatch to the correct SHAP explainer based on model type.

    Parameters
    ----------
    model_type : 'tree', 'linear', or 'auto' (auto-detects from clf class name)
    """
    if model_type == "auto":
        clf_name = ""
        if hasattr(pipeline, "named_steps"):
            clf_name = type(list(pipeline.named_steps.values())[-1]).__name__.lower()
        else:
            clf_name = type(pipeline).__name__.lower()

        if "xgb" in clf_name or "forest" in clf_name or "tree" in clf_name:
            model_type = "tree"
        else:
            model_type = "linear"

    if model_type == "tree":
        return compute_shap_tree(pipeline, X_train, X_explain, feature_names)
    else:
        return compute_shap_linear(pipeline, X_train, X_explain, feature_names)


def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature, sorted descending.

    Returns
    -------
    DataFrame with columns: feature, mean_abs_shap, rank
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Save SHAP values matrix to results/."""
    df = pd.DataFrame(shap_values, columns=feature_names)
    df.insert(0, "model", model_name)
    out_path = RESULTS_DIR / f"shap_values_{model_name.replace(' ', '_')}.csv"
    df.to_csv(out_path, index=False)
    print(f"[interpretability] Saved SHAP values to {out_path}")
    return df


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data_extraction import load_all_data
    from src.preprocessing import preprocess_expression_matrix, prepare_ml_arrays
    from src.models import get_model_definitions, get_endosomal_features

    data = load_all_data()
    matrix = preprocess_expression_matrix(data["mouse_matrix"])
    X, y, names = prepare_ml_arrays(matrix)

    endo = get_endosomal_features(names)
    endo_idx = [names.index(f) for f in endo if f in names]
    X_endo = X[:, endo_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_endo, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    models = get_model_definitions()
    xgb_pipe = models["Endosomal-XGB"]
    xgb_pipe.fit(X_tr, y_tr)

    shap_vals, ev = compute_shap_values(xgb_pipe, X_tr, X_te, endo)
    importance = shap_feature_importance(shap_vals, endo)
    print("[interpretability] Top 10 features by mean |SHAP|:")
    print(importance.head(10).to_string(index=False))
