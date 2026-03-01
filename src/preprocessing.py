"""
Proteomics preprocessing pipeline.

Steps (in order):
  1. Replace zeros with NaN  (zeros = below detection limit)
  2. Log2 transform if not already log-scale
  3. Filter proteins with >50% missing values
  4. KNN imputation (k=5) for remaining missing values
  5. Median centering per sample
  6. StandardScaler (fit on training data only — done inside CV loop)

Covariate regression (for human data only):
  CRITICAL: Fit on training fold, apply to test fold — never on full dataset.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.config import RANDOM_STATE


# ── Step helpers ──────────────────────────────────────────────────────────────

def is_log_scale(values: pd.Series | np.ndarray) -> bool:
    """Heuristic: log2-transformed LFQ values are typically in range 20–33."""
    arr = np.asarray(values, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return False
    return float(clean.max()) < 50


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(0, np.nan)


def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Log2-transform a DataFrame of non-negative raw intensities."""
    return df.apply(lambda col: np.log2(col.replace(0, np.nan)))


def filter_high_missing(df: pd.DataFrame, max_missing_frac: float = 0.5) -> pd.DataFrame:
    """Drop protein columns with fraction of NaN > max_missing_frac."""
    missing_frac = df.isna().mean(axis=0)
    keep = missing_frac[missing_frac <= max_missing_frac].index
    dropped = len(df.columns) - len(keep)
    if dropped > 0:
        print(f"[preprocessing] Dropped {dropped} proteins with >{max_missing_frac*100:.0f}% missing")
    return df[keep]


def knn_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """KNN imputation on a protein × sample matrix."""
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed = imputer.fit_transform(df.values)
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)


def median_center(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-sample median (each row is a sample)."""
    return df.subtract(df.median(axis=1), axis=0)


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_expression_matrix(
    df: pd.DataFrame,
    missing_threshold: float = 0.5,
    n_neighbors_knn: int = 5,
    already_log2: bool | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a samples × proteins DataFrame.

    The 'label' column is preserved (not transformed).

    Parameters
    ----------
    df                 : samples × proteins (with optional 'label' column)
    missing_threshold  : max fraction of NaN to keep a protein column
    n_neighbors_knn    : k for KNN imputation
    already_log2       : if None, auto-detected via is_log_scale()

    Returns
    -------
    Preprocessed DataFrame (samples × proteins), 'label' column intact.
    """
    label_col = df["label"].copy() if "label" in df.columns else None
    prot_df = df.drop(columns=["label"], errors="ignore")

    # Step 1: zeros → NaN
    prot_df = replace_zeros_with_nan(prot_df)

    # Step 2: log2 transform if needed
    if already_log2 is None:
        flat = prot_df.values.ravel()
        already_log2 = is_log_scale(flat[np.isfinite(flat)])

    if not already_log2:
        print("[preprocessing] Applying log2 transform")
        prot_df = log2_transform(prot_df)

    # Step 3: filter high-missing proteins
    prot_df = filter_high_missing(prot_df, max_missing_frac=missing_threshold)

    # Step 4: KNN imputation
    if prot_df.isna().any().any():
        prot_df = knn_impute(prot_df, n_neighbors=n_neighbors_knn)

    # Step 5: median centering per sample
    prot_df = median_center(prot_df)

    if label_col is not None:
        prot_df["label"] = label_col.values

    return prot_df


# ── Covariate regression (human data only) ───────────────────────────────────

def regress_covariates(
    protein_df: pd.DataFrame,
    covariate_df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Residualise protein levels against covariates.

    CRITICAL: Fit regression on TRAINING data only, apply to test.
    Call this inside the CV loop, not before splitting.

    Parameters
    ----------
    protein_df    : samples × proteins (log2-transformed, imputed)
    covariate_df  : samples × covariates (e.g. age, sex)
    covariates    : columns to use; defaults to all columns in covariate_df

    Returns
    -------
    residuals : DataFrame same shape as protein_df
    """
    if covariates is None:
        covariates = covariate_df.columns.tolist()
    cov = covariate_df[covariates].copy()

    residuals = protein_df.copy()
    for col in protein_df.columns:
        mask = protein_df[col].notna()
        if mask.sum() < len(covariates) + 2:
            continue  # not enough data to fit
        lr = LinearRegression()
        lr.fit(cov.loc[mask], protein_df.loc[mask, col])
        residuals.loc[mask, col] = (
            protein_df.loc[mask, col] - lr.predict(cov.loc[mask])
        )
    return residuals


# ── Build feature/label arrays for ML ────────────────────────────────────────

def prepare_ml_arrays(
    matrix: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract feature matrix X and label vector y from a preprocessed DataFrame.

    Parameters
    ----------
    matrix       : samples × proteins with 'label' column
    feature_cols : which protein columns to include; None = all except 'label'

    Returns
    -------
    X     : np.ndarray (n_samples, n_features)
    y     : np.ndarray (n_samples,)
    names : list of feature names
    """
    if feature_cols is None:
        feature_cols = [c for c in matrix.columns if c != "label"]
    feature_cols = [c for c in feature_cols if c in matrix.columns]

    X = matrix[feature_cols].values.astype(float)
    y = matrix["label"].values.astype(int)
    return X, y, feature_cols


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_extraction import load_all_data

    data = load_all_data()
    matrix = data["mouse_matrix"]
    print(f"[preprocessing] Raw mouse matrix shape: {matrix.shape}")
    processed = preprocess_expression_matrix(matrix)
    print(f"[preprocessing] After preprocessing:   {processed.shape}")
    print(f"  NaN remaining: {processed.drop(columns=['label']).isna().sum().sum()}")
    print(f"  Sample medians (should be ~0): {processed.drop(columns=['label']).median(axis=1).values.round(2)}")
