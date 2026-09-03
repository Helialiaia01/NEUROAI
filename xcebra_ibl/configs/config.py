"""
Configuration file for the xCEBRA-IBL pipeline.
All hyperparameters and paths are defined here.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# Kaggle mounts datasets read-only under /kaggle/input and only publishes files
# written under /kaggle/working. Keep local defaults compatible with the
# existing checkout while allowing the launcher to redirect generated output.
KAGGLE_INPUT_DIR = os.environ.get(
    "KAGGLE_INPUT_DIR", str(WORKSPACE_ROOT / "data" / "downloaded")
)
KAGGLE_WORKING_DIR = os.environ.get("KAGGLE_WORKING_DIR")

if KAGGLE_WORKING_DIR:
    OUT_DIR = Path(KAGGLE_WORKING_DIR)
    DATA_RAW_DIR = Path(KAGGLE_INPUT_DIR)
    DATA_PROCESSED_DIR = OUT_DIR / "data_processed"
    RESULTS_DIR = OUT_DIR / "results"
    MODELS_DIR = OUT_DIR / "trained_models"
else:
    OUT_DIR = WORKSPACE_ROOT
    DATA_RAW_DIR = Path(KAGGLE_INPUT_DIR)
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"  # preprocessed X, y
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = PROJECT_ROOT / "trained_models"

REPOS_DIR = Path(os.environ.get("XCEBRA_REPOS_DIR", str(WORKSPACE_ROOT / "repos")))

BRAINWIDE_RRR_REPO = REPOS_DIR / "brainwide-RRR"
ALLEN_AREA_LIST_CSV = BRAINWIDE_RRR_REPO / "example1" / "utils" / "area_list.csv"
ALLEN_CONN_MATRIX_CSV = BRAINWIDE_RRR_REPO / "example1" / "utils" / "conn_cxcx.csv"
RRR_RESULTS_DEFAULT = BRAINWIDE_RRR_REPO / "example1" / "trained_model_copy" / "RRRglobal_full.json"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────
# IBL Data Download Parameters  (matching brainwide-RRR)
# ──────────────────────────────────────────────────────
IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"
IBL_USERNAME = "intbrainlab"
IBL_PASSWORD = "international"

T_WINDOW = 1.2         # sec – total trial window
T_BF_STIMON = 0.3      # sec – time before stimulus onset
SPSDT = 10e-3           # sec – 10 ms time bin
MAX_SESSIONS_PER_AREA = 30

# 43 cortical areas from the paper
CORTICAL_AREAS = [
    "VISp", "AUDp", "SSp-ll", "AUDd", "SSp-n", "SSp-ul", "AIp",
    "SSp-m", "SSp-un", "SSp-bfd", "VISl", "AUDv", "SSs", "VISC",
    "SSp-tr", "VISli", "MOp", "VISrl", "VISpl", "RSPv", "RSPd",
    "GU", "RSPagl", "PERI", "ECT", "VISal", "ILA", "ORBl", "AId",
    "VISpm", "ORBm", "PL", "VISpor", "FRP", "AUDpo", "TEa", "VISa",
    "VISam", "MOs", "ORBvl", "ACAv", "ACAd", "AIv",
]

# ──────────────────────────────────────────────────────
# Preprocessing Parameters  (matching brainwide-RRR)
# ──────────────────────────────────────────────────────
# Neuron inclusion
MIN_FIRING_RATE = 0.5     # Hz
MAX_FIRING_RATE = 50.0    # Hz
MAX_SILENT_PROB = 0.5
MIN_NEURONS = 5
UNIT_LABEL_MIN = 0.0

# Trial inclusion
MIN_TRIALS = 100
REMOVE_BLOCK5 = True      # remove p(left)=0.5 trials

# Activity transform
GAUSSIAN_SMOOTH_SIGMA = 2.0  # bins (= 20 ms)
TRANSFORM_MFR = None          # None | "sqrt" | "log"
STANDARDIZE_Y = True
STANDARDIZE_X = True

# Areas to exclude
AREAS_EXCLUDE = ["root", "void", "y"]

# The 8 behavioral variables (in order)
VARIABLE_NAMES = [
    "block",           # prior probability block
    "side",            # stimulus side
    "contrast_level",  # stimulus contrast
    "choice",          # animal's choice
    "outcome",         # reward / error
    "wheel",           # wheel velocity (time-varying)
    "whisker_max",     # whisker motion energy (time-varying)
    "lick",            # lick rate (time-varying)
]
VARIABLE_DISPLAY_NAMES = [
    "Block", "Stimulus", "Contrast", "Choice",
    "Outcome", "Wheel", "Whisker", "Lick",
]
N_VARIABLES = len(VARIABLE_NAMES)

# ──────────────────────────────────────────────────────
# xCEBRA Model Hyperparameters
# ──────────────────────────────────────────────────────
# Embedding dimension per variable group (G groups, d_i dims each)
EMBEDDING_DIM_PER_GROUP = 4
TOTAL_EMBEDDING_DIM = N_VARIABLES * EMBEDDING_DIM_PER_GROUP  # 8 × 4 = 32

# CEBRA training
MODEL_ARCHITECTURE = "offset10-model"   # 10-step temporal context
MAX_ITERATIONS = 10000
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
TEMPERATURE = 1.0
NUM_HIDDEN_UNITS = 128
TIME_OFFSETS = 10

# Jacobian regularization (xCEBRA)
JACOBIAN_REG_WEIGHT = 0.01    # λ for ‖J‖² regularization
JACOBIAN_N_PROJ = 1           # number of random projections for Jacobian

# ──────────────────────────────────────────────────────
# Selectivity & Clustering Parameters
# ──────────────────────────────────────────────────────
MIN_DELTA_R2 = 0.015         # minimum ΔR² for neuron inclusion (RRR comparison)
MIN_NEURONS_PER_AREA = 20    # for selectivity profiles
MIN_NEURONS_PER_AREA_CORR = 50  # for connectivity correlation

# Cross-validation
N_CV_FOLDS = 5
TEST_FRACTION = 0.3
