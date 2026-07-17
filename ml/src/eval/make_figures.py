"""Thesis figures generated ONLY from experiment outputs (results/*.json).

No hand-entered numbers anywhere: every curve, bar and heatmap is read from
the JSON files that the training runs wrote. Grayscale + hatching so the
figures print correctly in black and white.

Outputs PNG (300 dpi) + PDF into results/figures/:
  curves_<id>            train/val loss + val macro-F1 per epoch (needs "history")
  confusion_<id>         test confusion-matrix heatmap
  model_comparison       test macro-F1 bars for baselines + proposed
  ablation               ablation test/val macro-F1 bars
  binary_comparison      binary pos/neg accuracy where recorded

Run from ml/:
    python -m src.eval.make_figures
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.schema import RESULTS_DIR

OUT = RESULTS_DIR / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5,
    "figure.dpi": 300,
})

MAIN_ORDER = ["tfidf_lr", "bilstm_fasttext", "mbert_full", "xlmr_full", "lora_scl_lid"]
MAIN_LABELS = {
    "tfidf_lr": "B1\nTF-IDF+LR", "bilstm_fasttext": "B2\nBiLSTM",
    "mbert_full": "B3\nmBERT full", "xlmr_full": "B4\nXLM-R full",
    "lora_scl_lid": "P1\nproposed",
}
ABL_ORDER = ["lora_scl_lid", "abl_no_scl", "abl_no_lid", "abl_lora_only",
             "abl_no_lora", "abl_rank_4", "abl_rank_16", "abl_q_only"]
ABL_LABELS = {
    "lora_scl_lid": "P1 (full)", "abl_no_scl": "no SCL", "abl_no_lid": "no LID",
    "abl_lora_only": "LoRA only", "abl_no_lora": "no LoRA\n(full FT)",
    "abl_rank_4": "rank 4", "abl_rank_16": "rank 16", "abl_q_only": "query only",
}
FILLS = ["0.95", "0.85", "0.75", "0.6", "0.45", "0.9", "0.7", "0.55"]
HATCH = ["", "//", "..", "xx", "\\\\", "", "//", ".."]


def load_all():
    out = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png/.pdf")


def fig_curves(rid, r):
    h = r["history"]
    ep = range(1, len(h["val_macro_f1"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a1.plot(ep, h["train_loss"], color="black", marker="o", ms=3, lw=1.2,
            label="training loss")
    a1.plot(ep, h["val_loss"], color="0.45", ls="--", marker="s", ms=3, lw=1.2,
            label="validation loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a1.legend(frameon=False, fontsize=7.5)
    a2.plot(ep, h["val_macro_f1"], color="black", marker="o", ms=3, lw=1.2)
    best = int(np.argmax(h["val_macro_f1"]))
    a2.scatter([best + 1], [h["val_macro_f1"][best]], facecolor="white",
               edgecolor="black", zorder=5, s=40)
    a2.annotate(f"best {h['val_macro_f1'][best]:.4f}", (best + 1, h["val_macro_f1"][best]),
                textcoords="offset points", xytext=(5, -10), fontsize=7.5)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation macro-F1")
    for a in (a1, a2):
        a.set_axisbelow(True)
    save(fig, f"curves_{rid}")


def fig_confusion(rid, r):
    cm = np.array(r["test"]["confusion_matrix"], dtype=float)
    labels = r["test"]["confusion_matrix_labels"]
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(row_pct, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if row_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(cm[i, j])}\n({row_pct[i, j]:.0%})",
                    ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.grid(False)
    save(fig, f"confusion_{rid}")


def _fmt_params(n):
    return f"{n/1e6:.2f}M" if n < 1e7 else f"{n/1e6:.0f}M"


def fig_model_comparison(res):
    ids = [i for i in MAIN_ORDER if i in res]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for k, rid in enumerate(ids):
        r = res[rid]
        b = ax.bar(k, r["test"]["macro_f1"], width=0.62, facecolor=FILLS[k],
                   hatch=HATCH[k], edgecolor="black", lw=0.8)
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"{MAIN_LABELS[i]}\n({_fmt_params(res[i]['trainable_params'])})"
                        for i in ids], fontsize=7.5)
    ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 0.8)
    ax.set_axisbelow(True); ax.grid(axis="x", visible=False)
    save(fig, "model_comparison")


def fig_ablation(res):
    ids = [i for i in ABL_ORDER if i in res]
    if len(ids) < 3:
        print("ablation figure skipped (JSONs not present yet)")
        return
    y = np.arange(len(ids))
    test = [res[i]["test"]["macro_f1"] for i in ids]
    val = [max(res[i]["history"]["val_macro_f1"]) if "history" in res[i]
           else res[i]["val"]["macro_f1"] for i in ids]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(y + 0.2, test, height=0.38, facecolor="0.4", edgecolor="black",
            lw=0.7, label="test")
    ax.barh(y - 0.2, val, height=0.38, facecolor="white", hatch="//",
            edgecolor="black", lw=0.7, label="validation (best)")
    for i, t in enumerate(test):
        ax.text(t + 0.003, y[i] + 0.2, f"{t:.3f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([ABL_LABELS.get(i, i) for i in ids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1"); ax.set_xlim(0.55, 0.78)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_axisbelow(True); ax.grid(axis="y", visible=False)
    save(fig, "ablation")


def fig_binary(res):
    ids = [i for i in ABL_ORDER + ["p1_r4_nolid", "p1_large"]
           if i in res and "extra" in res[i] and "binary_pos_neg" in res[i]["extra"]]
    if not ids:
        print("binary figure skipped (no runs carry binary metrics yet)")
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for k, rid in enumerate(ids):
        acc = res[rid]["extra"]["binary_pos_neg"]["accuracy"]
        b = ax.bar(k, acc, width=0.6, facecolor="0.8", edgecolor="black", lw=0.8)
        ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=2)
    ax.axhline(0.8, color="black", ls=":", lw=1)
    ax.text(len(ids) - 0.5, 0.802, "0.80 target", fontsize=7, ha="right")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([ABL_LABELS.get(i, i) for i in ids], fontsize=7.5)
    ax.set_ylabel("Binary pos/neg accuracy"); ax.set_ylim(0.5, 0.95)
    ax.set_axisbelow(True); ax.grid(axis="x", visible=False)
    save(fig, "binary_comparison")


def main():
    res = load_all()
    print("loaded:", ", ".join(sorted(res)))
    for rid, r in res.items():
        if "history" in r:
            fig_curves(rid, r)
    for rid in ("lora_scl_lid", "xlmr_full", "tfidf_lr"):
        if rid in res:
            fig_confusion(rid, res[rid])
    fig_model_comparison(res)
    fig_ablation(res)
    fig_binary(res)


if __name__ == "__main__":
    main()
