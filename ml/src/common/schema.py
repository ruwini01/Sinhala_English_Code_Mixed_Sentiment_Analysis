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
# Raw schema (11 columns, exact order — verified against the actual export;
# the research plan's 13-column description was outdated: this file has no
# sentiment_fine / label_source columns and names the clean column clean_text)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "id",
    "text",
    "sentiment_label",
    "source_platform",
    "source_url",
    "language_type",
    "clean_text",
    "text_length",
    "domain",
    "content_type",
    "emotion",
]

# Model inputs / target
TEXT_COL = "clean_text"
LID_COL = "language_type"
LABEL_COL = "sentiment_label"
ID_COL = "id"
PLATFORM_COL = "source_platform"

# Descriptive only — dataset-statistics tables, NEVER model inputs
DESCRIPTIVE_COLS = ["domain", "content_type", "emotion"]

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 3

# Rows whose sentiment_label is missing or not in LABEL2ID are quarantined,
# never remapped (verified in raw: 16 missing, 14 non-standard e.g.
# mixed/label/humorous/neutral-negative).
VALID_LABELS = set(LABEL2ID)

# Provenance values allowed in label_source (column ADDED by pipeline steps;
# not present in the raw file)
LABEL_SOURCES = {"manual", "auto", "deterministic"}

# ---------------------------------------------------------------------------
# Language ID tags (per-token, produced by src/preprocess/language_id.py)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-token language tag ids (produced by src/preprocess/tokenize_cache.py;
# consumed by the LID embedding layer). "special" covers <s>, </s>, <pad>.
# ---------------------------------------------------------------------------
LID_TAGS = {"sin": 0, "eng": 1, "other": 2, "special": 3}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
