"""Pipeline step 1: validate the raw dataset. NO MUTATION.

Asserts the 11-column schema, reports row counts per platform and per label,
flags quarantine candidates, and prints a DATA.md-ready provenance block
(sha256 + counts) to paste into ml/DATA.md.

Run from ml/:
    python -m src.preprocess.validate_schema

Exits non-zero if the schema does not match (wrong columns / unreadable file),
so it can gate the rest of the pipeline. Label/platform anomalies are REPORTED
but do not fail the run — separating those rows is step 2 (quarantine.py).
"""

import hashlib
import sys
from datetime import date

import pandas as pd

from src.common.schema import (
    ID_COL,
    LABEL_COL,
    PLATFORM_COL,
    RAW_COLUMNS,
    RAW_CSV,
    TEXT_COL,
    VALID_LABELS,
)


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not RAW_CSV.exists():
        sys.exit(f"ERROR: raw file not found: {RAW_CSV}")

    digest = sha256_of(RAW_CSV)
    # dtype=str + no NA coercion of values: read as close to byte-exact as
    # csv parsing allows; utf-8-sig strips the BOM the export carries.
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str)

    print(f"file:   {RAW_CSV}")
    print(f"sha256: {digest}")
    print(f"rows:   {len(df)}")
    print(f"cols:   {len(df.columns)}")

    if list(df.columns) != RAW_COLUMNS:
        print("\nSCHEMA MISMATCH")
        print(f"  expected: {RAW_COLUMNS}")
        print(f"  found:    {list(df.columns)}")
        sys.exit(1)
    print("schema: OK (11 columns, exact order)")

    # --- integrity report (informational, handled by later steps) ---------
    n_dup_ids = int(df[ID_COL].duplicated().sum())
    n_empty_text = int(df["text"].isna().sum())
    n_empty_clean = int(df[TEXT_COL].isna().sum())

    labels = df[LABEL_COL].str.strip().str.lower()
    valid_mask = labels.isin(VALID_LABELS)
    n_missing_label = int(df[LABEL_COL].isna().sum())
    n_nonstandard = int((~valid_mask).sum() - n_missing_label)

    print("\n--- integrity ---")
    print(f"duplicate ids:            {n_dup_ids}")
    print(f"missing text:             {n_empty_text}")
    print(f"missing {TEXT_COL}:       {n_empty_clean}")
    print(f"missing {LABEL_COL}:      {n_missing_label}")
    print(f"non-standard labels:      {n_nonstandard}")
    print(f"3-class-ready rows:       {int(valid_mask.sum())}")

    print("\n--- per label (raw values) ---")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())

    print("\n--- per platform (raw values) ---")
    print(df[PLATFORM_COL].value_counts(dropna=False).to_string())

    # --- DATA.md block -----------------------------------------------------
    print("\n--- paste into ml/DATA.md under 'Raw input' ---")
    print(f"- sha256:   {digest}")
    print(f"- rows:     {len(df)} total; {int(valid_mask.sum())} 3-class-ready; "
          f"{n_missing_label} missing label; {n_nonstandard} non-standard label")
    print(f"- verified: {date.today().isoformat()} by src/preprocess/validate_schema.py")


if __name__ == "__main__":
    main()
