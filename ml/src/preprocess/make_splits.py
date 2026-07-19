"""Pipeline step 5: GROUP-AWARE stratified train/val/test split (70/15/15, seed=42).

Input : data/interim/lid.csv            (step 4 output; never modified)
Output: data/processed/splits/train_ids.txt
        data/processed/splits/val_ids.txt
        data/processed/splits/test_ids.txt
        data/processed/splits/excluded_conflict_ids.txt
        (one row id per line — these files are COMMITTED to git)

Leakage protection (v3):
  1. Texts are normalized (NFC, lowercase, punctuation/emoji stripped,
     whitespace collapsed) and identical normalized texts form a GROUP.
  2. Groups whose rows carry CONFLICTING sentiment labels are EXCLUDED from
     all splits (annotation noise, quarantined by id, never trained on).
  3. The split is stratified over GROUPS, so every duplicate text lands in
     exactly one split — zero normalized-text overlap, verified at the end.

Splits are stratified on sentiment_label. Once written, the files are LOCKED:
this script refuses to overwrite them. Never re-split.

Run from ml/:
    python -m src.preprocess.make_splits
"""

import hashlib
import re
import subprocess
import sys
import unicodedata
from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.schema import (
    ID_COL,
    INTERIM_DIR,
    LABEL_COL,
    SEED,
    SPLITS_DIR,
    TEXT_COL,
)

IN_CSV = INTERIM_DIR / "lid.csv"
OUT_FILES = {name: SPLITS_DIR / f"{name}_ids.txt" for name in ("train", "val", "test")}
EXCLUDED_FILE = SPLITS_DIR / "excluded_conflict_ids.txt"

NON_WORD_RE = re.compile(r"[^\w\s඀-෿]")
WS_RE = re.compile(r"\s+")


def normalize_text(text):
    """Duplicate-detection normalization: NFC, lowercase, no punct/emoji."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = NON_WORD_RE.sub("", t)
    return WS_RE.sub(" ", t).strip()


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
    df["_ntext"] = df[TEXT_COL].map(normalize_text)

    # empty-normalized rows (emoji-only) can't collide meaningfully — make
    # each its own group via its id
    df.loc[df["_ntext"] == "", "_ntext"] = "<empty:" + df[ID_COL] + ">"

    # 1. exclude conflicting-label duplicate groups
    n_labels = df.groupby("_ntext")[LABEL_COL].transform("nunique")
    conflict = df[n_labels > 1]
    df_ok = df[n_labels == 1]
    n_conflict_groups = conflict["_ntext"].nunique()

    # 2. one row per group, stratified 70/15/15 over groups
    groups = df_ok.groupby("_ntext").agg(
        label=(LABEL_COL, "first"), n_rows=(ID_COL, "size")).reset_index()
    g_train, g_temp = train_test_split(
        groups, test_size=0.30, random_state=SEED, stratify=groups["label"]
    )
    g_val, g_test = train_test_split(
        g_temp, test_size=0.50, random_state=SEED, stratify=g_temp["label"]
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, gpart in (("train", g_train), ("val", g_val), ("test", g_test)):
        part = df_ok[df_ok["_ntext"].isin(gpart["_ntext"])]
        splits[name] = part
        OUT_FILES[name].write_text("\n".join(part[ID_COL]) + "\n", encoding="utf-8")
    EXCLUDED_FILE.write_text(
        "\n".join(conflict[ID_COL]) + ("\n" if len(conflict) else ""), encoding="utf-8")

    # 3. verify: no row lost, ZERO normalized-text overlap between splits
    assert sum(len(p) for p in splits.values()) + len(conflict) == len(df)
    texts = {n: set(p["_ntext"]) for n, p in splits.items()}
    assert not texts["train"] & texts["val"], "train/val text overlap!"
    assert not texts["train"] & texts["test"], "train/test text overlap!"
    assert not texts["val"] & texts["test"], "val/test text overlap!"

    print(f"total: {len(df)}  usable: {len(df_ok)}  "
          f"excluded (conflicting-label duplicates): {len(conflict)} rows "
          f"in {n_conflict_groups} groups -> {EXCLUDED_FILE.name}")
    for name, part in splits.items():
        dist = part[LABEL_COL].value_counts(normalize=True).round(3).to_dict()
        print(f"{name:5s}: {len(part):5d} rows  label share: {dist}")
    print("verified: zero normalized-text overlap between train/val/test")

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/processed/splits/{train,val,test}_ids.txt")
    print(f"- source:   data/interim/lid.csv (sha256: {sha256_of(IN_CSV)[:16]}...)")
    print(f"- script:   src/preprocess/make_splits.py @ git commit {git_head()}")
    sizes = ", ".join(f"{n} {len(p)}" for n, p in splits.items())
    print(f"- output:   {sizes}; {len(conflict)} conflict rows excluded "
          "(committed to git; LOCKED, never re-split)")
    print(f"- date:     {date.today().isoformat()}")
    print("- rationale: GROUP-AWARE stratified 70/15/15 on sentiment_label, seed=42; "
          "duplicate texts share a split, conflicting-label duplicates quarantined, "
          "zero normalized-text overlap verified")


if __name__ == "__main__":
    main()