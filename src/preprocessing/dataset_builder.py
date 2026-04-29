"""
Dataset builder for Sinhala-English code-mixed sentiment analysis.

Reads the collected dataset (singlish_mixed_sentiment_labeled.csv),
runs the full preprocessing pipeline, splits into train/val/test,
and computes class weights for imbalanced training.

Usage:
    python -m src.preprocessing.dataset_builder \
        --input  "path/to/singlish_mixed_sentiment_labeled.csv" \
        --outdir "data/processed"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.cleaner import clean_text
from src.preprocessing.lid_tagger import tag_and_enrich

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
VALID_SENTIMENTS = set(LABEL_MAP.keys())
VALID_PLATFORMS = {"youtube", "facebook"}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    # drop header-repeated rows (where sentiment == 'sentiment' or 'label')
    df = df[df["sentiment"].str.lower().isin(VALID_SENTIMENTS)]
    df = df[df["platform"].str.lower().isin(VALID_PLATFORMS)]
    # normalise
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df["platform"] = df["platform"].str.lower().str.strip()
    # drop rows with missing text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def run_preprocessing(df: pd.DataFrame, demojize: bool = False) -> pd.DataFrame:
    print(f"  Cleaning {len(df)} texts …")
    df["cleaned_text"] = df["text"].apply(lambda t: clean_text(t, demojize=demojize))

    # drop anything that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print(f"  After cleaning: {len(df)} rows remain")

    print("  Running LID tagging …")
    lid_results = df["cleaned_text"].apply(tag_and_enrich)
    lid_df = pd.json_normalize(lid_results)

    df["tagged_text"] = lid_df["tagged_text"].values
    df["lid_sequence"] = lid_df["lid_sequence"].values
    df["n_sinhala"] = lid_df["n_sinhala"].values
    df["n_english"] = lid_df["n_english"].values
    df["n_emoji"] = lid_df["n_emoji"].values
    df["cmi"] = lid_df["cmi"].values
    df["is_code_mixed"] = lid_df["is_code_mixed"].values

    df["label"] = df["sentiment"].map(LABEL_MAP)
    return df


def split_dataset(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                  random_state: int = 42):
    """Stratified split: 70 / 15 / 15."""
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df["label"], random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_weights(train_df: pd.DataFrame) -> dict:
    labels = train_df["label"].values
    classes = np.array(sorted(LABEL_MAP.values()))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): round(float(w), 4) for c, w in zip(classes, weights)}


def save_splits(train_df, val_df, test_df, outdir: str, weights: dict):
    os.makedirs(outdir, exist_ok=True)
    save_cols = ["id", "text", "cleaned_text", "tagged_text", "lid_sequence",
                 "sentiment", "label", "platform", "source",
                 "n_sinhala", "n_english", "n_emoji", "cmi", "is_code_mixed"]

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cols = [c for c in save_cols if c in split_df.columns]
        split_df[cols].to_csv(os.path.join(outdir, f"{split_name}.csv"), index=False, encoding="utf-8-sig")
        print(f"  Saved {split_name}.csv  ({len(split_df)} rows)")

    weights_path = os.path.join(outdir, "class_weights.txt")
    with open(weights_path, "w") as f:
        f.write("# class_weights for CrossEntropyLoss\n")
        f.write("# label: 0=positive  1=negative  2=neutral\n")
        for cls, w in weights.items():
            label_name = {v: k for k, v in LABEL_MAP.items()}[cls]
            f.write(f"{cls} ({label_name}): {w}\n")
    print(f"  Saved class_weights.txt → {weights}")


def build_dataset(input_path: str, outdir: str, demojize: bool = False):
    print("\n[1/4] Loading raw data …")
    df = load_raw(input_path)
    print(f"  Loaded {len(df)} clean rows | sentiment dist:\n{df['sentiment'].value_counts().to_dict()}")

    print("\n[2/4] Preprocessing …")
    df = run_preprocessing(df, demojize=demojize)

    print("\n[3/4] Splitting (70/15/15) …")
    train_df, val_df, test_df = split_dataset(df)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print("\n[4/4] Computing class weights …")
    weights = compute_weights(train_df)
    print(f"  Weights: {weights}")

    print("\nSaving splits …")
    save_splits(train_df, val_df, test_df, outdir, weights)
    print("\nDone.")
    return train_df, val_df, test_df, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--demojize", action="store_true",
                        help="Convert emojis to text descriptions (for BiLSTM/TF-IDF)")
    args = parser.parse_args()
    build_dataset(args.input, args.outdir, demojize=args.demojize)
