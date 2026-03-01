"""
Shared utilities: results tables, metadata writing, saving helpers.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.config import RESULTS_DIR, RANDOM_STATE
from src.training import compare_models_paired


def generate_comparison_table(
    results: dict,
    reference_model: str = "Simoes-3",
) -> pd.DataFrame:
    """
    Generate a summary comparison table suitable for a manuscript.

    Columns:
      Model | N_Features | AUC | AUC_95CI | N_Folds | Delta_AUC | P_vs_Reference
    """
    rows = []
    for name, res in results.items():
        row = {
            "Model":      name,
            "N_Features": res.get("n_features", "?"),
            "AUC":        f"{res['mean_auc']:.3f}",
            "AUC_95CI":   f"({res['ci_95_low']:.3f}–{res['ci_95_high']:.3f})",
            "N_Folds":    res["n_folds"],
        }
        if name != reference_model and reference_model in results:
            comp = compare_models_paired(results, reference_model, name)
            row["Delta_AUC"]      = f"{comp['delta_auc']:+.3f}"
            row["P_vs_Reference"] = f"{comp['p_value']:.4f}"
        else:
            row["Delta_AUC"]      = "—"
            row["P_vs_Reference"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results: dict, filename: str = "mouse_classification_results.csv") -> None:
    """Save per-fold AUC results to CSV."""
    rows = []
    for name, res in results.items():
        for fold_i, auc in enumerate(res["aucs"]):
            rows.append({"model": name, "fold": fold_i, "auc": auc})
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / filename
    df.to_csv(out, index=False)
    print(f"[utils] Saved {out}")


def save_stability_results(
    selection_probs: np.ndarray,
    feature_names: list[str],
    filename: str = "stability_selection_results.csv",
) -> pd.DataFrame:
    df = pd.DataFrame({
        "protein":             feature_names,
        "selection_frequency": selection_probs,
    }).sort_values("selection_frequency", ascending=False).reset_index(drop=True)
    out = RESULTS_DIR / filename
    df.to_csv(out, index=False)
    print(f"[utils] Saved {out}")
    return df


def write_metadata(metadata: dict, filename: str = "run_metadata.json") -> None:
    """Write pipeline run metadata to results/."""
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_state": RANDOM_STATE,
        **metadata,
    }
    out = RESULTS_DIR / filename
    with open(out, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[utils] Saved metadata to {out}")


def describe_results(results: dict) -> None:
    """Print a human-readable summary of model comparison results."""
    print("\n" + "=" * 65)
    print(f"{'Model':<26} {'AUC':>7} {'± SD':>7} {'95% CI':>18} {'n_feat':>7}")
    print("-" * 65)
    for name, res in results.items():
        print(
            f"{name:<26} {res['mean_auc']:>7.3f} {res['std_auc']:>7.3f} "
            f"({res['ci_95_low']:.3f}–{res['ci_95_high']:.3f}) "
            f"{res.get('n_features', '?'):>7}"
        )
    print("=" * 65)
