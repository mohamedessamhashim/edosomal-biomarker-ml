"""
Central configuration for the endosomal biomarker ML pipeline.
ALL paths, constants, and hyperparameters live here.
"""
from pathlib import Path
import os

# ── Detect Environment ──
IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
if IN_COLAB:
    PROJECT_ROOT = Path("/content/endosomal-biomarker-ml")
else:
    PROJECT_ROOT = Path(__file__).parent.parent

# ── Paths ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for _d in [RAW_DATA_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Data Mode ──
DATA_MODE = "auto"  # "supplement", "placeholder", "auto"

# ── Supplement Filenames ──
S1_FILENAME = "aba6334_data_file_s1.xlsx"
S3_FILENAME = "aba6334_data_file_s3.xlsx"

# ── Simoes 2020 Reference ──
SIMOES_PMC_ID = "PMC7901670"
SIMOES_DOI = "10.1126/scitranslmed.aba6334"
N_TOTAL_PROTEINS = 1505
N_SIGNIFICANT_PROTEINS = 71
N_PARAMETRIC = 52
N_NONPARAMETRIC = 19
SIGNIFICANCE_THRESHOLD = 0.05

# ── Known Validated Proteins ──
SIMOES_VALIDATED_3 = ["Aplp1", "Chl1", "Mapt"]
SIMOES_VALIDATED_3_HUMAN = ["APLP1", "CHL1", "MAPT"]

# ── ML Hyperparameters ──
RANDOM_STATE = 42
CV_N_SPLITS = 5
CV_N_REPEATS = 10
STABILITY_N_ITERATIONS = 500       # Reduce to 200 for Colab / inside-CV
STABILITY_N_ITERATIONS_CV = 200    # Used inside each CV fold
STABILITY_SAMPLE_FRACTION = 0.7    # Reduce to 0.5 for small mouse data
STABILITY_SAMPLE_FRACTION_SMALL = 0.5
STABILITY_THRESHOLD = 0.6
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_N_ESTIMATORS = 200
ELASTICNET_L1_RATIO = 0.5
ELASTICNET_C = 0.1
ELASTICNET_MAX_ITER = 10000

# ── Synthetic Data ──
SYNTHETIC_N_CONTROL = 250
SYNTHETIC_N_MCI = 200
SYNTHETIC_N_AD = 150
SYNTHETIC_NOISE_SCALE = 0.3

# ── Figure Settings ──
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf"]

# ── Ensembl API ──
ENSEMBL_REST_SERVER = "https://rest.ensembl.org"
ENSEMBL_RATE_LIMIT_SLEEP = 0.07  # seconds between requests

# ── g:Profiler ──
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
GPROFILER_ORGANISM_MOUSE = "mmusculus"
GPROFILER_ORGANISM_HUMAN = "hsapiens"

# ── STRING-db ──
STRING_URL = "https://string-db.org/api/tsv/network"
STRING_SPECIES_MOUSE = 10090
STRING_SPECIES_HUMAN = 9606
STRING_SCORE_THRESHOLD = 400
