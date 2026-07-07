"""Standard results contract: every experiment writes results/<experiment_id>.json.

All entrypoints in src/train/ call save_results() so that every run records
the same fields — metrics, efficiency numbers (trainable params, wall-clock),
provenance (git commit, date, seed) — and src/eval/make_tables.py can build
the thesis tables from results/*.json without special cases.
"""

import json
import subprocess
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.common.schema import ID2LABEL, RESULTS_DIR, SEED


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def evaluate(y_true, y_pred):
    """Full metric block for one split. Labels are ints 0/1/2."""
    classes = sorted(ID2LABEL)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            ID2LABEL[c]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_matrix_labels": [ID2LABEL[c] for c in classes],
    }


def save_results(experiment_id, val_metrics, test_metrics, trainable_params,
                 wall_clock_sec, hyperparams, notes=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment_id": experiment_id,
        "git_commit": git_head(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "trainable_params": int(trainable_params),
        "wall_clock_sec": round(float(wall_clock_sec), 1),
        "hyperparams": hyperparams,
        "val": val_metrics,
        "test": test_metrics,
        "notes": notes,
    }
    path = RESULTS_DIR / f"{experiment_id}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nresults -> {path}")
    print(f"test macro-F1: {test_metrics['macro_f1']:.4f}  "
          f"accuracy: {test_metrics['accuracy']:.4f}  "
          f"trainable params: {trainable_params:,}  "
          f"wall clock: {wall_clock_sec:.0f}s")
    return path