"""Schema constants for the Singlish sentiment dataset.

Single source of truth for column names, label mappings and canonical paths.
Every pipeline step imports from here — never redefine these locally.
See ml/DATA.md for provenance of the raw file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to ml/ — all scripts are run from ml/ as `python -m src...`)
# ---------------------------------------------------------------------------
ML_ROOT = Path(__file__).resolve().parents[2]          # .../ml
DATA_DIR = ML_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "singlish_mixed_sentiment_complete.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"
RESULTS_DIR = ML_ROOT / "results"
DATA_MD = ML_ROOT / "DATA.md"

# ---------------------------------------------------------------------------
# Raw schema (13 columns, exact order)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "id",
    "text",
    "sentiment_label",
    "source_platform",
    "source_url",
    "language_type",
    "text_clean",
    "text_length",
    "domain",
    "content_type",
    "emotion",
    "sentiment_fine",
    "label_source",
]

# Model inputs / target
TEXT_COL = "text_clean"
LID_COL = "language_type"
LABEL_COL = "sentiment_label"
ID_COL = "id"

# Descriptive only — dataset-statistics tables, NEVER model inputs
DESCRIPTIVE_COLS = ["domain", "content_type", "emotion", "sentiment_fine"]

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 3

# Rows whose sentiment_label is missing or not in LABEL2ID are quarantined,
# never remapped (known: ~16 missing, ~30 non-standard e.g. mixed/humorous).
VALID_LABELS = set(LABEL2ID)

# Provenance values allowed in label_source
LABEL_SOURCES = {"manual", "auto", "deterministic"}

# ---------------------------------------------------------------------------
# Language ID tags (per-token, produced by src/preprocess/language_id.py)
# ---------------------------------------------------------------------------
LID_TAGS = {"sin": 0, "eng": 1, "mixed": 2, "unknown": 3}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
