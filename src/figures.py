"""
Publication-quality figure generation.

Figures produced:
  Fig 1  — Volcano plot (fold change vs -log10 p-value)
  Fig 2  — ROC comparison for all 5 models
  Fig 3A — SHAP beeswarm (top 20 proteins)
  Fig 3B — Mean |SHAP| bar chart
  Fig 4  — Stability selection bar chart
  Fig 5  — GO enrichment dot plot
  Fig S1 — Correlation heatmap of endosomal proteins
  Fig S2 — Forest plot of AUC ± CI

Global style set via set_publication_style().
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve

from src.config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMATS, RESULTS_DIR


# ── Colour palette ────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "Simoes-3":           "#3498db",
    "Random-3":           "#95a5a6",
    "Endosomal-Full":     "#e67e22",
    "Endosomal-Optimized": "#e74c3c",
    "Endosomal-XGB":      "#2ecc71",
}

SIGNIFICANCE_COLORS = {
    "significant_up":   "#e74c3c",
    "significant_down": "#3498db",
    "not_significant":  "#cccccc",
    "validated_3":      "#f1c40f",
}


# ── Global style ──────────────────────────────────────────────────────────────

def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        FIGURE_DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth":     0.8,
        "lines.linewidth":    1.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    for fmt in FIGURE_FORMATS:
        path = FIGURES_DIR / f"{stem}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[figures] Saved {stem}")


# ── Fig 1: Volcano plot ────────────────────────────────────────────────────────

def plot_volcano(
    protein_df: pd.DataFrame,
    validated_genes: list[str] | None = None,
    output_stem: str = "fig1_volcano",
) -> None:
    """
    Volcano plot: log2(FC) on x-axis, -log10(p) on y-axis.

    protein_df must have columns: mouse_gene, log2_fc, pvalue, direction, analysis_type
    """
    if validated_genes is None:
        validated_genes = ["Aplp1", "Chl1", "Mapt"]

    df = protein_df.copy()
    df["-log10_p"] = -np.log10(df["pvalue"].clip(1e-10))

    sig_mask   = df["analysis_type"].isin(["parametric", "nonparametric"])
    val_mask   = df["mouse_gene"].str.lower().isin([g.lower() for g in validated_genes])

    colors = np.where(
        val_mask,
        SIGNIFICANCE_COLORS["validated_3"],
        np.where(
            sig_mask & (df["direction"] == "up"),
            SIGNIFICANCE_COLORS["significant_up"],
            np.where(
                sig_mask & (df["direction"] == "down"),
                SIGNIFICANCE_COLORS["significant_down"],
                SIGNIFICANCE_COLORS["not_significant"],
            )
        )
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["log2_fc"], df["-log10_p"], c=colors, s=12, alpha=0.7, linewidths=0)
    ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="gray", alpha=0.6)
    ax.axvline(0, ls="--", lw=0.5, color="gray", alpha=0.4)

    # Annotate validated 3
    for _, row in df[val_mask].iterrows():
        ax.annotate(
            row["mouse_gene"], (row["log2_fc"], row["-log10_p"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=8, fontweight="bold",
        )

    patches = [
        mpatches.Patch(color=SIGNIFICANCE_COLORS["significant_up"],   label="Significant ↑"),
        mpatches.Patch(color=SIGNIFICANCE_COLORS["significant_down"],  label="Significant ↓"),
        mpatches.Patch(color=SIGNIFICANCE_COLORS["validated_3"],       label="Simoes validated"),
        mpatches.Patch(color=SIGNIFICANCE_COLORS["not_significant"],   label="Not significant"),
    ]
    ax.legend(handles=patches, loc="upper left", frameon=True, framealpha=0.9)
    ax.set_xlabel("log₂ Fold Change (VPS35-cKO / Control)")
    ax.set_ylabel("-log₁₀ p-value")
    ax.set_title("CSF Proteomics: VPS35-cKO vs Control")
    _save(fig, output_stem)
    plt.close(fig)


# ── Fig 2: ROC comparison ─────────────────────────────────────────────────────

def plot_roc_comparison(
    results: dict,
    output_stem: str = "fig2_roc_comparison",
    data_source: str = "",
) -> None:
    """
    Overlaid ROC curves for all models with AUC ± SD in legend.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results.items():
        if not len(res.get("all_y_true", [])):
            continue
        y_true = res["all_y_true"]
        y_prob = res["all_y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = MODEL_COLORS.get(name, "gray")
        label = f"{name} (AUC={res['mean_auc']:.3f} ± {res['std_auc']:.3f})"
        ax.plot(fpr, tpr, color=color, label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    title = "Model Comparison: Endosomal Biomarker Classification"
    if data_source:
        title += f"\n[{data_source}]"
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _save(fig, output_stem)
    plt.close(fig)


# ── Fig 3A: SHAP beeswarm ────────────────────────────────────────────────────

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    output_stem: str = "fig3a_shap_beeswarm",
) -> None:
    import shap

    fig, ax = plt.subplots(figsize=(8, max_display * 0.35 + 1))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Values: Top Protein Contributions")
    _save(plt.gcf(), output_stem)
    plt.close("all")


# ── Fig 3B: SHAP bar chart ────────────────────────────────────────────────────

def plot_shap_bar(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_stem: str = "fig3b_shap_bar",
) -> None:
    """
    Horizontal bar chart of mean |SHAP| values.

    importance_df: output of shap_feature_importance()
    """
    top = importance_df.head(top_n).iloc[::-1]  # reverse for bottom-up bars
    fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 1))
    colors = [MODEL_COLORS["Endosomal-Optimized"] if i < 3 else "#aaaaaa"
              for i in range(len(top))][::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    _save(fig, output_stem)
    plt.close(fig)


# ── Fig 4: Stability selection ────────────────────────────────────────────────

def plot_stability_selection(
    selection_probs: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.6,
    top_n: int = 30,
    output_stem: str = "fig4_stability_selection",
) -> None:
    idx = np.argsort(selection_probs)[-top_n:]
    probs = selection_probs[idx]
    names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, top_n * 0.3 + 1))
    colors = [MODEL_COLORS["Endosomal-Optimized"] if p >= threshold else "#cccccc"
              for p in probs]
    ax.barh(names, probs, color=colors, edgecolor="none")
    ax.axvline(threshold, ls="--", lw=1.2, color="black", alpha=0.7,
               label=f"Threshold = {threshold}")
    ax.set_xlabel("Selection Probability")
    ax.set_title("Stability Selection: Protein Selection Frequencies")
    ax.legend(loc="lower right")
    _save(fig, output_stem)
    plt.close(fig)


# ── Fig 5: GO enrichment dot plot ────────────────────────────────────────────

def plot_go_enrichment(
    enrichment_df: pd.DataFrame,
    top_n: int = 15,
    output_stem: str = "fig5_go_enrichment",
) -> None:
    if enrichment_df.empty:
        print("[figures] GO enrichment DataFrame is empty — skipping Fig 5")
        return

    name_col  = next((c for c in ["term_name", "name", "description"] if c in enrichment_df.columns), None)
    pval_col  = next((c for c in ["p_value", "p.value", "pvalue", "adjusted_p_value"] if c in enrichment_df.columns), None)
    count_col = next((c for c in ["intersection_size", "overlap_size", "count"] if c in enrichment_df.columns), None)

    if name_col is None or pval_col is None:
        print("[figures] GO enrichment columns not found — skipping Fig 5")
        return

    top = enrichment_df.nsmallest(top_n, pval_col).copy()
    top["-log10_p"] = -np.log10(top[pval_col].clip(1e-10))
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    scatter_size = top[count_col] * 20 if count_col else 100
    sc = ax.scatter(top["-log10_p"], range(len(top)), s=scatter_size,
                    c=top["-log10_p"], cmap="YlOrRd", edgecolors="gray", linewidths=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[name_col], fontsize=8)
    ax.set_xlabel("-log₁₀ p-value")
    ax.set_title("GO Term Enrichment (ML-selected proteins)")
    plt.colorbar(sc, ax=ax, label="-log₁₀ p")
    _save(fig, output_stem)
    plt.close(fig)


# ── Fig S1: Correlation heatmap ───────────────────────────────────────────────

def plot_correlation_clustermap(
    matrix: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_stem: str = "figS1_correlation_clustermap",
) -> None:
    df = matrix.drop(columns=["label"], errors="ignore")
    if feature_cols:
        df = df[[c for c in feature_cols if c in df.columns]]

    corr = df.T.corr()
    g = sns.clustermap(
        corr, cmap="RdBu_r", vmin=-1, vmax=1,
        figsize=(max(6, len(corr) * 0.35), max(6, len(corr) * 0.35)),
        linewidths=0.3, annot=len(corr) <= 20,
    )
    g.fig.suptitle("Endosomal Protein Correlation Clustermap", y=1.02)
    for fmt in FIGURE_FORMATS:
        g.fig.savefig(FIGURES_DIR / f"{output_stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[figures] Saved {output_stem}")
    plt.close("all")


# ── Fig S2: Forest plot ───────────────────────────────────────────────────────

def plot_model_comparison_forest(
    results: dict,
    output_stem: str = "figS2_forest_plot",
) -> None:
    model_names = list(results.keys())
    means  = [results[n]["mean_auc"]   for n in model_names]
    lows   = [results[n]["ci_95_low"]  for n in model_names]
    highs  = [results[n]["ci_95_high"] for n in model_names]
    err_lo = [m - l for m, l in zip(means, lows)]
    err_hi = [h - m for m, h in zip(means, highs)]

    fig, ax = plt.subplots(figsize=(6, len(model_names) * 0.8 + 1))
    y_pos = range(len(model_names))
    colors = [MODEL_COLORS.get(n, "#666666") for n in model_names]
    ax.errorbar(
        means, y_pos,
        xerr=[err_lo, err_hi],
        fmt="o", markersize=8, capsize=5,
        color="black", ecolor="gray",
    )
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter(m, i, s=80, color=c, zorder=5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(model_names)
    ax.axvline(0.5, ls="--", lw=0.8, color="gray", alpha=0.6)
    ax.set_xlabel("AUC (95% CI)")
    ax.set_title("Model Comparison: AUC with 95% Confidence Intervals")
    ax.set_xlim(0.3, 1.0)
    _save(fig, output_stem)
    plt.close(fig)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_extraction import load_all_data
    from src.preprocessing import preprocess_expression_matrix, prepare_ml_arrays
    from src.training import run_model_comparison

    set_publication_style()
    data = load_all_data()
    matrix = preprocess_expression_matrix(data["mouse_matrix"])
    X, y, names = prepare_ml_arrays(matrix)

    print("[figures] Running quick training for figure demo …")
    results = run_model_comparison(
        X, y, names,
        cv_config={"n_splits": 5, "n_repeats": 2, "seed": 42},
        n_stability_iter=30,
        verbose=False,
    )

    plot_volcano(data["sig_proteins"])
    plot_roc_comparison(results, data_source="Simoes 2020 (placeholder)")
    plot_model_comparison_forest(results)
    print(f"[figures] All figures saved to {FIGURES_DIR}")
