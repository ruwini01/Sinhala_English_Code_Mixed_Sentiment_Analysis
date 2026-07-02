"""
Shared helpers for all models (baselines B1-B4 and the proposed model).

Keeps three things consistent across every experiment so the numbers are
comparable:
  1. how the train/val/test splits are loaded,
  2. which column is the model input and which is the label,
  3. how predictions are scored (Macro F1 is the PRIMARY metric, because it
     weights all three classes equally regardless of how imbalanced they are).

The richer plotting / statistical-significance code lives later in
src/evaluation/ — this file is deliberately light (sklearn + numpy only)
so even the CPU-only B1 baseline can import it without pulling in torch.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# label id -> human name. Must match LABEL_MAP in preprocessing/dataset_builder.py
LABEL_NAMES = ["positive", "negative", "neutral"]  # index = label id (0,1,2)

PROCESSED_DIR = "data/processed"
TEXT_COL = "cleaned_text"   # model input
LABEL_COL = "label"         # integer target (0/1/2)


def load_split(split: str, processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load one of 'train' / 'val' / 'test' as a DataFrame."""
    path = os.path.join(processed_dir, f"{split}.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def get_xy(df: pd.DataFrame, text_col: str = TEXT_COL):
    """Return (texts, labels) as a list[str] and an int numpy array."""
    texts = df[text_col].astype(str).tolist()
    labels = df[LABEL_COL].to_numpy()
    return texts, labels


def evaluate(y_true, y_pred) -> dict:
    """Score predictions. Macro F1 is the headline number for this thesis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2], target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in LABEL_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def format_report(metrics: dict, title: str = "") -> str:
    """Pretty-print a metrics dict as an aligned text block."""
    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
    lines.append(f"Accuracy    : {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1    : {metrics['macro_f1']:.4f}   <-- PRIMARY")
    lines.append(f"Weighted F1 : {metrics['weighted_f1']:.4f}")
    lines.append("")
    lines.append(f"{'class':<10}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}")
    for name in LABEL_NAMES:
        c = metrics["per_class"][name]
        lines.append(f"{name:<10}{c['precision']:>11.3f}{c['recall']:>9.3f}{c['f1']:>8.3f}{c['support']:>9}")
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred) [positive, negative, neutral]:")
    for name, row in zip(LABEL_NAMES, metrics["confusion_matrix"]):
        lines.append(f"  {name:<9}{row}")
    return "\n".join(lines)


def save_metrics(metrics: dict, path: str) -> None:
    """Write a metrics dict to JSON (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
