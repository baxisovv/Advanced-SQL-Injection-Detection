import logging
import sys
import warnings
import os
from pathlib import Path

# --- Paths ---
# Support environment variable override
DATA_PATH = os.getenv('SQLI_DATA_PATH', 'rbsqli_dataset.csv')
MODEL_SAVE_DIR = './models_out/'

# --- Dataset Validation ---
def validate_dataset():
    """Validate dataset exists before processing"""
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Please download the RBSQLi dataset from Mendeley "
            f"or set SQLI_DATA_PATH environment variable."
        )
    return True

# --- Sampling & Split Constraints ---
# Now configurable via environment variables
MAX_BENIGN = int(os.getenv('MAX_BENIGN', 500000))
MAX_MALICIOUS = int(os.getenv('MAX_MALICIOUS', 100000))
TEST_SIZE_RATIO = 0.20   
VAL_SIZE_RATIO  = 0.20   
TEST_PRIOR_BENIGN    = 0.95
TEST_PRIOR_MALICIOUS = 0.05

# --- Adaptive Sampling Based on Available Memory ---
def adaptive_sampling():
    """Adjust sampling based on available system memory"""
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:
            return 50000, 10000  # Reduced for low-memory systems
        elif available_memory_gb < 8:
            return 200000, 50000
        else:
            return MAX_BENIGN, MAX_MALICIOUS
    except ImportError:
        # If psutil not available, use configured values
        return MAX_BENIGN, MAX_MALICIOUS

# --- Feature Engineering ---
# TF-IDF aggressively constrained to force models to rely on structural logic
TFIDF_MAX_FEATURES = 150 
NGRAM_RANGE        = (3, 5)

# --- Chi-Square Feature Selection Configuration ---
ENABLE_CHI_SQUARE_SELECTION = True
CHI_SQUARE_K_FEATURES = None  # None = auto-determine, or set to specific number
CHI_SQUARE_PERCENTILE = 75    # Keep top 75% of features by chi-square score
CHI_SQUARE_P_VALUE_THRESHOLD = 0.05  # Statistical significance threshold

# --- Undersampling Configuration ---
UNDERSAMPLING_STRATEGY = 'random'  # Options: 'random', 'tomek', 'nearmiss', 'hybrid', None
UNDERSAMPLING_RATIO = 1.0  # 1.0 = balanced, 0.5 = 2:1 majority:minority

# --- Model Selection ---
TUNE_HYPERPARAMS = True

# --- Cross-Validation ---
CV_N_SPLITS = 5   

# --- Suppress specific library warnings ---
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Logging Configuration ---
def setup_logger():
    logger = logging.getLogger('SQLi_Pipeline')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        fh = logging.FileHandler('pipeline.log', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = setup_logger()
