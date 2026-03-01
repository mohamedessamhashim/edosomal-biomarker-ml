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

## Reproducibility

All random seeds fixed at `42`. Results are deterministic across runs.
Synthetic data is clearly labeled in all outputs and figures.
