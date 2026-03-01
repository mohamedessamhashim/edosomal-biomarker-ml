"""
Generate synthetic human-scale CSF proteomics data.

Calibrated to published effect sizes from Simoes et al. 2020 so that ML
models achieve realistic AUCs (~0.80–0.90 for the endosomal panel).

All outputs are clearly labelled as synthetic.  This module is used to
demonstrate the pipeline at realistic sample sizes (n=600) when real
large-scale human data is not yet available.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RANDOM_STATE, PROCESSED_DIR,
    SYNTHETIC_N_CONTROL, SYNTHETIC_N_MCI, SYNTHETIC_N_AD,
    SYNTHETIC_NOISE_SCALE,
)

# Metadata stamped onto every synthetic output
SYNTHETIC_METADATA = {
    "data_source": "synthetic",
    "calibration": "Published fold changes from Simoes et al. 2020 Sci Transl Med",
    "n_samples":   SYNTHETIC_N_CONTROL + SYNTHETIC_N_MCI + SYNTHETIC_N_AD,
    "generation_seed": RANDOM_STATE,
    "WARNING": (
        "This is synthetic data for pipeline demonstration. "
        "Results do NOT represent real clinical findings."
    ),
}


def generate_synthetic_human_cohort(
    protein_stats_df: pd.DataFrame,
    n_control: int = SYNTHETIC_N_CONTROL,
    n_mci: int    = SYNTHETIC_N_MCI,
    n_ad: int     = SYNTHETIC_N_AD,
    noise_scale: float = SYNTHETIC_NOISE_SCALE,
    seed: int          = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a synthetic CSF/proteomics dataset for simulated AD clinical trial subjects.

    Effect sizes per disease group (log2 scale relative to control):
      Control : mean = 0
      MCI     : mean = 0.3 × log2_fc   (partial effect)
      AD      : mean = log2_fc          (full published effect)

    Biological noise is added as Gaussian with SD = noise_scale × disease_modifier.

    Parameters
    ----------
    protein_stats_df : DataFrame with columns [mouse_gene, log2_fc, analysis_type]
                       (output of generate_placeholder_proteins or from real data)
    n_control / n_mci / n_ad : cohort sizes
    noise_scale : baseline SD for log2 protein abundance
    seed        : random seed

    Returns
    -------
    df : pd.DataFrame  shape (n_total, n_proteins + 5)
        Columns: sample_id, diagnosis (0/1/2), age, sex, apoe4, <protein_cols...>
    """
    rng = np.random.default_rng(seed)
    n_total = n_control + n_mci + n_ad
    labels  = [0] * n_control + [1] * n_mci + [2] * n_ad

    # ── Covariates ──
    ages = np.concatenate([
        rng.normal(75, 8, n_control),
        rng.normal(76, 7, n_mci),
        rng.normal(78, 7, n_ad),
    ]).clip(50, 95)

    sex = rng.choice([0, 1], size=n_total, p=[0.45, 0.55])  # 55% female

    apoe4_probs = (
        [0.25] * n_control + [0.40] * n_mci + [0.55] * n_ad
    )
    apoe4 = np.array([rng.binomial(1, p) for p in apoe4_probs])

    # ── Protein expression ──
    protein_data = {}
    for _, row in protein_stats_df.iterrows():
        gene    = row["mouse_gene"]
        log2_fc = float(row.get("log2_fc", 0.0))
        is_sig  = row.get("analysis_type", "background") in ("parametric", "nonparametric")

        vals = []
        for lbl in labels:
            if is_sig:
                if lbl == 0:    mean, sd = 0.0,              noise_scale
                elif lbl == 1:  mean, sd = 0.3 * log2_fc,    noise_scale * 1.1
                else:           mean, sd = log2_fc,           noise_scale * 1.2
            else:
                mean, sd = 0.0, noise_scale
            vals.append(rng.normal(mean, sd))
        protein_data[gene] = vals

    df = pd.DataFrame(protein_data)
    df.insert(0, "sample_id",  [f"SYN_{i:04d}" for i in range(1, n_total + 1)])
    df.insert(1, "diagnosis",  labels)
    df.insert(2, "age",        ages.round(1))
    df.insert(3, "sex",        sex)            # 0=M, 1=F
    df.insert(4, "apoe4",      apoe4)

    return df


def make_binary_human_dataset(
    full_df: pd.DataFrame,
    groups: tuple[int, int] = (0, 2),   # Control vs AD by default
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Subset to two diagnostic groups and return (X, y).

    Parameters
    ----------
    full_df : output of generate_synthetic_human_cohort
    groups  : tuple of two diagnosis codes to keep

    Returns
    -------
    X : DataFrame of protein columns only (no covariates)
    y : Series of binary labels (0 = groups[0], 1 = groups[1])
    """
    sub = full_df[full_df["diagnosis"].isin(groups)].copy()
    covariate_cols = ["sample_id", "diagnosis", "age", "sex", "apoe4"]
    protein_cols   = [c for c in sub.columns if c not in covariate_cols]
    X = sub[protein_cols].reset_index(drop=True)
    y = (sub["diagnosis"] == groups[1]).astype(int).reset_index(drop=True)
    return X, y


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_extraction import load_all_data

    data = load_all_data()
    sig_df = data["sig_proteins"]

    print("[synthetic_data] Generating 600-subject synthetic human cohort …")
    synth = generate_synthetic_human_cohort(sig_df)

    print(f"  Shape:       {synth.shape}")
    print(f"  Diagnosis counts:\n{synth['diagnosis'].value_counts().sort_index()}")
    print(f"  Age mean:    {synth['age'].mean():.1f} ± {synth['age'].std():.1f}")
    print(f"  APOE4 rate:  {synth['apoe4'].mean():.2f}")

    # Quick sanity check on APLP1 effect size
    aplp1_ctrl = synth.loc[synth["diagnosis"] == 0, "Aplp1"].mean()
    aplp1_ad   = synth.loc[synth["diagnosis"] == 2, "Aplp1"].mean()
    print(f"\n  Aplp1 log2-FC (AD vs Ctrl): {aplp1_ad - aplp1_ctrl:.2f}  (expected ~1.23)")

    out_path = PROCESSED_DIR / "synthetic_human_data.csv"
    synth.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
