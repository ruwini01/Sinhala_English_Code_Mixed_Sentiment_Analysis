"""Pipeline step 2: separate 3-class-ready rows from quarantined rows.

Input : data/raw/singlish_mixed_sentiment_complete.csv  (never modified)
Output: data/interim/clean.csv       — rows with sentiment_label in
                                       {negative, neutral, positive}
                                       (label normalized: strip + lowercase)
        data/interim/quarantine.csv  — everything else, with an added
                                       quarantine_reason column; labels kept
                                       verbatim, NEVER remapped

Run from ml/:
    python -m src.preprocess.quarantine

Prints a DATA.md-ready provenance block. Fails if clean + quarantine row
counts do not reconcile with the raw total.
"""

import hashlib
import subprocess
import sys
from datetime import date

import pandas as pd

from src.common.schema import (
    INTERIM_DIR,
    LABEL_COL,
    PLATFORM_COL,
    RAW_COLUMNS,
    RAW_CSV,
    VALID_LABELS,
)

KNOWN_PLATFORMS = {"youtube", "facebook", "tiktok", "google_play"}


def normalize_platform(row):
    """Canonical lowercase platform; a few raw rows have language values
    shifted into source_platform, so fall back to inferring from source_url."""
    v = str(row[PLATFORM_COL]).strip().lower().replace(" ", "_")
    if v in KNOWN_PLATFORMS:
        return v
    url = str(row["source_url"]).strip().lower()
    for p in KNOWN_PLATFORMS:
        if p.replace("_", "") in url.replace("_", "").replace(".", ""):
            return p
    return "unknown"

CLEAN_CSV = INTERIM_DIR / "clean.csv"
QUARANTINE_CSV = INTERIM_DIR / "quarantine.csv"


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str)
    if list(df.columns) != RAW_COLUMNS:
        sys.exit("ERROR: schema mismatch — run validate_schema first")

    n_raw = len(df)
    df[PLATFORM_COL] = df.apply(normalize_platform, axis=1)
    normalized = df[LABEL_COL].str.strip().str.lower()
    valid_mask = normalized.isin(VALID_LABELS)

    clean = df[valid_mask].copy()
    clean[LABEL_COL] = normalized[valid_mask]  # canonical lowercase form

    quarantine = df[~valid_mask].copy()
    quarantine["quarantine_reason"] = [
        "missing_label" if pd.isna(v) else f"non_standard_label:{v}"
        for v in df.loc[~valid_mask, LABEL_COL]
    ]

    if len(clean) + len(quarantine) != n_raw:
        sys.exit(
            f"ERROR: row counts do not reconcile: "
            f"{len(clean)} + {len(quarantine)} != {n_raw}"
        )

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8", lineterminator="\n")
    quarantine.to_csv(QUARANTINE_CSV, index=False, encoding="utf-8", lineterminator="\n")

    print(f"raw:        {n_raw}")
    print(f"clean:      {len(clean)}  -> {CLEAN_CSV}")
    print(f"quarantine: {len(quarantine)}  -> {QUARANTINE_CSV}")
    print("\nclean label counts:")
    print(clean[LABEL_COL].value_counts().to_string())
    print("\nquarantine reasons:")
    print(quarantine["quarantine_reason"].value_counts().to_string())

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/interim/clean.csv + data/interim/quarantine.csv")
    print(f"- source:   data/raw/singlish_mixed_sentiment_complete.csv "
          f"(sha256: {sha256_of(RAW_CSV)[:16]}...)")
    print(f"- script:   src/preprocess/quarantine.py @ git commit {git_head()}")
    print(f"- output:   clean.csv {len(clean)} rows "
          f"(sha256: {sha256_of(CLEAN_CSV)[:16]}...); "
          f"quarantine.csv {len(quarantine)} rows "
          f"(sha256: {sha256_of(QUARANTINE_CSV)[:16]}...)")
    print(f"- date:     {date.today().isoformat()}")
    print("- rationale: separate 3-class-ready rows; quarantined labels kept "
          "verbatim, never remapped")


if __name__ == "__main__":
    main()
