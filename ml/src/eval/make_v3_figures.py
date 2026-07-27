"""Generate the Chapter 4 figures from the v3 result files.

Every value plotted here is read from a results JSON produced by a training
run. Nothing is typed in by hand, so the figures cannot drift away from the
numbers reported in the thesis tables.

Run from the ml/ directory:
    python -X utf8 -m src.eval.make_v3_figures
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- where the numbers come from and where the pictures go -----------------
SRC = Path(r"C:\Users\Ruwini Tharanga\Downloads\thesis_v3_final_results")
OUT = Path(__file__).resolve().parents[2] / "results" / "figures"
THESIS = Path(r"C:\Users\Ruwini Tharanga\Downloads\IT4216_Research_Thesis_Template\figures")

# --- pastel palette, matching the dataset figures already in the thesis ----
GREEN, RED, BLUE = "#a8d5ba", "#f4a9a8", "#a8c8e8"
ORANGE, PURPLE, GREY = "#f6c99f", "#c9b8e0", "#cccccc"
INK = "#333333"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def load(name: str) -> dict:
    with open(SRC / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    if THESIS.exists():
        shutil.copy2(path, THESIS / f"{stem}.png")
    print(f"  wrote {path.name}")


# ===========================================================================
# Figure 10 - test macro-F1 of every model
# ===========================================================================
def fig_model_comparison():
    order = [
        ("tfidf_lr", "B1\nTF-IDF+LR", GREY),
        ("bilstm_fasttext", "B2\nBiLSTM", GREY),
        ("mbert_full", "B3\nmBERT full", GREY),
        ("xlmr_full", "B4\nXLM-R full", GREY),
        ("lora_scl_lid", "P1\nproposed", BLUE),
        ("p1_large", "P1-large", GREEN),
    ]
    names, scores, colours = [], [], []
    for key, label, colour in order:
        names.append(label)
        scores.append(load(key)["test"]["macro_f1"])
        colours.append(colour)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(names, scores, color=colours, edgecolor="white", width=0.62)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.006, f"{s:.3f}",
                ha="center", fontsize=9.5)

    best = max(scores)
    ax.axhline(best, color="#b0b0b0", linestyle="--", linewidth=1)
    ax.text(1.55, best + 0.0035, "best single system",
            fontsize=8.5, color="#777777", ha="left")

    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0.55, max(scores) + 0.045)
    ax.set_title("Test macro-F1 on the held-out set of 1,525 comments",
                 fontsize=11, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_10_model_comparison")


# ===========================================================================
# Figure 11 - training and validation loss, the overfitting evidence
# ===========================================================================
def fig_training_curves():
    runs = [
        ("abl_no_lora", "Full fine-tuning", RED),
        ("lora_scl_lid", "P1 proposed", BLUE),
        ("p1_large", "P1-large", GREEN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (key, title, colour) in zip(axes, runs):
        h = load(key)["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=colour, linewidth=2,
                marker="o", markersize=3.5, label="training loss")
        ax.plot(ep, h["val_loss"], color=colour, linewidth=2,
                linestyle="--", marker="s", markersize=3.5,
                alpha=0.75, label="validation loss")
        gap = h["val_loss"][-1] - h["train_loss"][-1]
        ax.fill_between(ep, h["train_loss"], h["val_loss"],
                        color=colour, alpha=0.13)
        ax.set_title(f"{title}\nfinal validation loss {abs(gap):.2f} "
                     f"{'above' if gap > 0 else 'below'} training loss",
                     fontsize=9.5)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Loss")
    fig.suptitle("Training and validation loss by epoch",
                 fontsize=11, y=1.05)
    save(fig, "fig_11_training_curves")


# ===========================================================================
# Figure 12 - confusion matrix of the proposed model
# ===========================================================================
def fig_confusion():
    d = load("lora_scl_lid")["test"]
    cm = np.array(d["confusion_matrix"], dtype=float)
    labels = [s.capitalize() for s in d["confusion_matrix_labels"]]
    recall = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues",
              vmin=0, vmax=1, alpha=0.72)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{int(cm[i, j])}\n{share:.0%}", ha="center",
                    va="center", fontsize=10,
                    color="white" if share > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)),
                  [f"{l}\n(recall {r:.1%})" for l, r in zip(labels, recall)])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix of the proposed model", fontsize=11, pad=10)
    ax.grid(False)
    save(fig, "fig_12_confusion_matrix")


# ===========================================================================
# Figure 13 - ablation
# ===========================================================================
def fig_ablation():
    variants = [
        ("abl_no_lora", "Full fine-tune\n(no LoRA)"),
        ("abl_rank_4", "Rank 4"),
        ("abl_no_scl", "Without SCL"),
        ("lora_scl_lid", "P1\n(rank 8, all parts)"),
        ("abl_no_lid", "Without LID"),
        ("abl_lora_only", "LoRA only"),
        ("abl_rank_16", "Rank 16"),
        ("abl_q_only", "Query only"),
    ]
    base = load("lora_scl_lid")["test"]["macro_f1"]
    names = [lbl for _, lbl in variants]
    deltas = [load(k)["test"]["macro_f1"] - base for k, _ in variants]
    colours = [BLUE if k == "lora_scl_lid" else (GREEN if d >= 0 else RED)
               for (k, _), d in zip(variants, deltas)]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.barh(names, deltas, color=colours, edgecolor="white", height=0.62)
    ax.axvline(0, color="#888888", linewidth=1)
    # the noise band the thesis declares
    ax.axvspan(-0.01, 0.01, color="#dddddd", alpha=0.5, zorder=0)
    lo, hi = min(deltas) - 0.011, max(deltas) + 0.009
    ax.set_xlim(lo, hi)
    # the left half of the top rows is empty, so the note sits there
    ax.text(lo + 0.0015, 0.75,
            "shaded band: differences of this size\n"
            "lie within run-to-run variation",
            fontsize=8, color="#777777", va="center", ha="left")
    for bar, d in zip(bars, deltas):
        off = 0.0012 if d >= 0 else -0.0012
        ax.text(d + off, bar.get_y() + bar.get_height() / 2, f"{d:+.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Change in test macro-F1 against the proposed model")
    ax.set_title("Effect of removing or changing one component at a time",
                 fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_13_ablation")


# ===========================================================================
# Figure 14 - three-class against the positive/negative subtask
# ===========================================================================
def fig_binary():
    variants = [
        ("abl_no_lora", "Full fine-tune"),
        ("p1_large", "P1-large"),
        ("abl_rank_4", "Rank 4"),
        ("abl_no_scl", "Without SCL"),
        ("lora_scl_lid", "P1 proposed"),
        ("abl_no_lid", "Without LID"),
        ("abl_lora_only", "LoRA only"),
        ("abl_rank_16", "Rank 16"),
        ("abl_q_only", "Query only"),
    ]
    names, three, binar = [], [], []
    for key, label in variants:
        d = load(key)
        b = d.get("extra", {}).get("binary_pos_neg", {})
        acc = b.get("accuracy") or b.get("acc")
        if acc is None:
            continue
        names.append(label)
        three.append(d["test"]["accuracy"])
        binar.append(acc)

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(y - 0.2, binar, height=0.38, color=GREEN, edgecolor="white",
            label="positive vs negative only (1,062 comments)")
    ax.barh(y + 0.2, three, height=0.38, color=BLUE, edgecolor="white",
            label="all three classes (1,525 comments)")
    for i, (b, t) in enumerate(zip(binar, three)):
        ax.text(b + 0.006, i - 0.2, f"{b:.1%}", va="center", fontsize=8.5)
        ax.text(t + 0.006, i + 0.2, f"{t:.1%}", va="center", fontsize=8.5)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.93)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy on the two-class polarity subtask and on the "
                 "full three-class task", fontsize=11, pad=34)
    # legend sits above the axes so it cannot cover a bar or a value label
    ax.legend(fontsize=8.5, frameon=False, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_14_binary_subtask")


if __name__ == "__main__":
    print(f"reading results from {SRC}")
    print("generating figures:")
    fig_model_comparison()
    fig_training_curves()
    fig_confusion()
    fig_ablation()
    fig_binary()
    print(f"\nsaved to {OUT}")
    if THESIS.exists():
        print(f"copied to {THESIS}")
