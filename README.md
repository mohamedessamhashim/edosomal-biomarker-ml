# Endosomal Biomarker ML Pipeline

Machine learning analysis of CSF proteomics from Simoes et al. 2020 (*Science Translational Medicine*) to identify endosomal biomarkers for Alzheimer's disease.

## Scientific Background

Simoes et al. conditionally knocked out VPS35 (a core retromer subunit) in mouse neurons and collected CSF for label-free quantification mass spectrometry. This pipeline:

1. Extracts quantitative protein data from the published supplementary materials
2. Maps mouse → human orthologs
3. Trains five nested ML classifiers (VPS35-cKO vs control)
4. Applies stability selection to independently recover APLP1, CHL1, and tau
5. Identifies additional endosomal biomarker candidates
6. Validates against real human clinical data from the same paper

**Citation:** Simoes S, et al. "Tau and other proteins found in Alzheimer's disease spinal fluid are linked to retromer-mediated endosomal traffic in mice and humans." *Sci Transl Med* 12(571):eaba6334, 2020. [DOI: 10.1126/scitranslmed.aba6334](https://doi.org/10.1126/scitranslmed.aba6334)

## Data Sources

| File | Contents |
|------|---------|
| `aba6334_data_file_s1.xlsx` | 1,505 detected proteins (Accession + Symbol) |
| `aba6334_data_file_s3.xlsx` | Per-sample quantitative data: LFQ intensities (Fig. 2), MS/MS peak intensities (Fig. S1), real human CSF clinical data (Fig. 7) |

## Quick Start (macOS / Linux)

```bash
git clone https://github.com/mohamedessamhashim/endosomal-biomarker-ml.git
cd endosomal-biomarker-ml

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run individual modules
python -m src.data_extraction
python -m src.training
python -m src.figures

# Or run notebooks in order
jupyter notebook notebooks/
```

## Google Colab

Open `colab/endosomal_biomarker_pipeline.ipynb` in [Google Colab](https://colab.research.google.com) and click **Runtime → Run all**. The notebook is fully self-contained.

## Project Structure

```
src/                  Python source modules
notebooks/            Step-by-step Jupyter notebooks (01–07)
colab/                Self-contained Colab notebook
data/raw/             Supplement files (place here)
data/processed/       Pipeline outputs (CSV)
figures/              Generated figures (PNG + PDF)
results/              Model comparison tables (CSV)
```

## Models

| Model | Features | Purpose |
|-------|----------|---------|
| Simoes-3 | APLP1, CHL1, MAPT | Baseline (published panel) |
| Random-3 | 3 random non-endosomal | Negative control |
| Endosomal-Full | ~60 endosomal proteins | Elastic net on full panel |
| Endosomal-Optimized | Stability-selected subset | Main result |
| Endosomal-XGB | Stability-selected subset | Nonlinear interactions |

## Preliminary Results

The pipeline evaluates five models across synthetic and mouse datasets using nested cross-validation. Performance is measured using the Area Under the Receiver Operating Characteristic Curve (AUC).

The extraordinarily high AUC scores (often reaching 1.0) observed in these preliminary results warrant careful interpretation. We provide the following detailed explanation for these findings:
- **High Dimensionality (p >> n)**: The datasets contain significantly more features than samples, mathematically guaranteeing linear separability and making perfect classification trivial.
- **Selection Bias in Baselines**: The Simoes-3 model utilizes exactly the biomarkers previously discovered in this specific dataset, introducing inherent circular evaluation.
- **Synthetic Data Artifacts**: The synthetic data generation process may preserve or amplify class separation characteristics, rendering the synthetic classification tasks artificially simple.
- **Over-Parameterization**: Models utilizing a large number of features (e.g., Endosomal-Full) perfectly memorize the limited training subset, leading to classic overfitting.
- **Small Sample Sizes**: With very few biological replicates available, even nested cross-validation struggles to provide truly independent validation, leading to over-optimistic AUC estimates.
- **Extreme Biological Phenotypes**: The mouse knock-out models represent an artificially induced, homogenous biological state that is far easier to classify than natural, heterogeneous disease progression.
- **Feature Selection Bias**: Identifying the "best" subset of features on small datasets before final evaluation can upwardly bias the apparent performance of the optimized models.
- **Lack of External Validation**: The perfect AUC scores reflect internal dataset characteristics rather than generalizable diagnostic utility, highlighting the absolute need for validation on independent clinical cohorts.

### Synthetic Data Performance

| Model | Features | AUC (95% CI) |
|-------|----------|--------------|
| Simoes-3 | 3 | 1.000 (0.999–1.000) |
| Random-3 | 3 | 0.962 (0.933–0.984) |
| Endosomal-Full | 14 | 1.000 (1.000–1.000) |
| Endosomal-Optimized | 3 | 0.999 (0.991–1.000) |
| Endosomal-XGB | 3 | 0.998 (0.986–1.000) |

### Mouse Data Performance

| Model | Features | AUC (95% CI) |
|-------|----------|--------------|
| Simoes-3 | 3 | 1.000 (1.000–1.000) |
| Random-3 | 3 | 0.625 (0.000–1.000) |
| Endosomal-Full | 27 | 0.500 (0.500–0.500) |
| Endosomal-Optimized | 3 | 0.988 (0.988–1.000) |
| Endosomal-XGB | 3 | 0.500 (0.500–0.500) |

## Reproducibility

All random seeds fixed at `42`. Results are deterministic across runs.
Synthetic data is clearly labeled in all outputs and figures.

## Limitations

- **Species Translation**: Mapping mouse knock-out models to human orthologs may not fully capture human-specific Alzheimer's disease pathology or temporal dynamics.
- **Sample Size and Overfitting**: Proteomics datasets typically feature high dimensionality with limited sample sizes, increasing the risk of overfitting despite the use of stability selection and regularization.
- **Synthetic Data Usage**: While synthetic data aids in pipeline development, it may not perfectly reflect the biological noise and complex covariance structures present in real clinical samples.
- **External Validation**: Findings represent biomarker *candidates* and require extensive validation in larger, independent, and longitudinal human clinical cohorts before diagnostic utility can be established.
