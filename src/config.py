"""
config.py
Purpose: keep EVERY system parameter in one place (requirement NFR4).
         The other modules only read from here and never hard-code values.
"""
from pathlib import Path

# ---------------- Paths ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"

# ---------------- Algorithm ----------------
MEASURE = "cosine"         # "cosine" | "pearson" | "jaccard" (measure used by the web app)
MODE = "item"              # "item" = Item-based CF (main); "user" is only used for comparison
K_NEIGHBORS = 50           # neighbours kept per movie (k); None = keep all
BETA = 10                  # shrinkage threshold on shared raters (beta); 0 = no shrinkage
MIN_ITEM_RATINGS = 10      # a movie needs at least m ratings to be recommended

RATING_MIN = 0.5
RATING_MAX = 5.0

# ---------------- Evaluation ----------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
EVAL_K = 5                 # list length for Precision@K and Recall@K
RELEVANT_THRESHOLD = 4.0   # a test movie rated >= this threshold is "relevant"

# ---------------- Web application ----------------
TOP_N_DEFAULT = 10
TOP_N_MIN = 2
TOP_N_MAX = 20
PROFILE_LIMIT = 60         # number of profiles offered for selection
WATCHED_DEFAULT = 10


def centered_for(measure: str) -> bool:
    """Pearson uses the mean-centred prediction formula (3.6); the other measures use (3.5)."""
    return measure == "pearson"
