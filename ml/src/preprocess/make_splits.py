"""Pipeline step 5: stratified train/val/test split (70/15/15, seed=42).

Input : data/interim/lid.csv            (step 4 output; never modified)
Output: data/processed/splits/train_ids.txt
        data/processed/splits/val_ids.txt
        data/processed/splits/test_ids.txt
        (one row id per line — these files are COMMITTED to git)

Splits are stratified on sentiment_label. Once written, the files are LOCKED:
this script refuses to overwrite them. Never re-split.

Run from ml/:
    python -m src.preprocess.make_splits
"""

import hashlib
import subprocess
import sys
from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.schema import (
    ID_COL,
    INTERIM_DIR,
    LABEL_COL,
    SEED,
    SPLITS_DIR,
)

IN_CSV = INTERIM_DIR / "lid.csv"
OUT_FILES = {name: SPLITS_DIR / f"{name}_ids.txt" for name in ("train", "val", "test")}


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
    existing = [str(p) for p in OUT_FILES.values() if p.exists()]
    if existing:
        sys.exit(
            "ERROR: split files already exist — splits are locked, never re-split:\n  "
            + "\n  ".join(existing)
        )

    df = pd.read_csv(IN_CSV, encoding="utf-8", dtype=str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[LABEL_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[LABEL_COL]
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in splits.items():
        OUT_FILES[name].write_text("\n".join(part[ID_COL]) + "\n", encoding="utf-8")

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    print(f"total: {len(df)}")
    for name, part in splits.items():
        dist = part[LABEL_COL].value_counts(normalize=True).round(3).to_dict()
        print(f"{name:5s}: {len(part):5d} rows  label share: {dist}")

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/processed/splits/{train,val,test}_ids.txt")
    print(f"- source:   data/interim/lid.csv (sha256: {sha256_of(IN_CSV)[:16]}...)")
    print(f"- script:   src/preprocess/make_splits.py @ git commit {git_head()}")
    sizes = ", ".join(f"{n} {len(p)}" for n, p in splits.items())
    print(f"- output:   {sizes} (committed to git; LOCKED, never re-split)")
    print(f"- date:     {date.today().isoformat()}")
    print("- rationale: stratified 70/15/15 on sentiment_label, seed=42")


if __name__ == "__main__":
    main()