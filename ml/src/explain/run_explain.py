"""Explainability stage (claim 2): SHAP + lexicon aggregation + faithfulness.

Runs on the trained P1 checkpoint. Everything is computed from the model and
the locked test split; outputs are JSON/CSV plus grayscale figures, all
regenerable by re-running this script.

Steps:
  1. SHAP word-level attributions on N sampled test comments (Partition
     explainer with a whitespace text masker).
  2. Sinhala lexicon aggregation: mean |attribution| for intensifiers and
     negations, spelling-variant-aware.
  3. Faithfulness test: remove each sample's top-5 attributed words, measure
     the macro-F1 drop (random-5 removal as control).
  4. Minimal pairs: prediction + top words for hand-picked sentence pairs
     that differ by one Sinhala word.

Run from ml/ (Colab T4, needs: pip install shap; checkpoint in checkpoints/):
    python -m src.explain.run_explain --ckpt checkpoints/lora_scl_lid.pt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.common.schema import ID2LABEL, LABEL2ID, LID_TAGS, RESULTS_DIR, SEED
from src.models.proposed_xlmr_lora import ProposedModel
from src.preprocess.language_id import token_lang
from src.train.data_io import load_split_frames
from src.train.results_io import evaluate

OUT = RESULTS_DIR / "explain"
FIGS = RESULTS_DIR / "figures"
MAX_LEN = 128

# Variant-aware Sinhala sentiment lexicon (claim 2). Latin-script spelling
# variants observed in the corpus; extend freely — matching is exact on
# lowercased alphabetic-stripped words.
LEXICON = {
    "intensifier": {
        "harima": ["harima", "harima", "harimaa"],
        "godak": ["godak", "godk", "goodak", "godaak"],
        "hari": ["hari", "haari"],
    },
    "negation": {
        "ne": ["ne", "ne.", "neh", "nee", "nae", "නෑ", "නැ"],
        "nehe": ["nehe", "nehee", "naha", "nehemai", "නැහැ"],
        "epa": ["epa", "eppa", "එපා"],
        "na": ["na", "naa", "nA"],
    },
}

MINIMAL_PAIRS = [
    ("film eka hodai", "film eka hodai ne"),
    ("me phone eka harima hodai", "me phone eka hodai"),
    ("service eka godak hondai", "service eka hondai"),
    ("mama meka kamathi", "mama meka kamathi ne"),
    ("ela machan supiri", "epa machan meka ganna"),
]


def norm_word(w):
    return re.sub(r"[^a-z඀-෿]", "", w.lower())


class P1Wrapper:
    """Raw text list -> class probabilities, rebuilding LID tags on the fly."""

    def __init__(self, ckpt, model_name, device):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = ProposedModel(model_name=model_name)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device

    def __call__(self, texts):
        texts = [str(t) if str(t).strip() else "…" for t in texts]
        words_list = [t.split() for t in texts]
        enc = self.tok(words_list, is_split_into_words=True, truncation=True,
                       max_length=MAX_LEN, padding=True, return_tensors="pt")
        lid = torch.full_like(enc["input_ids"], LID_TAGS["special"])
        for i, words in enumerate(words_list):
            tags = [LID_TAGS.get(token_lang(w), LID_TAGS["other"]) for w in words]
            for pos, wid in enumerate(enc.word_ids(batch_index=i)):
                if wid is not None:
                    lid[i, pos] = tags[wid]
        probs = []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                logits, _ = self.model(enc["input_ids"][s:s+64].to(self.device),
                                       enc["attention_mask"][s:s+64].to(self.device),
                                       lid[s:s+64].to(self.device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs)


def main():
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lora_scl_lid.pt")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    f = P1Wrapper(args.ckpt, args.model, device)
    test = load_split_frames()["test"]
    rng = np.random.default_rng(SEED)
    sample = test.iloc[rng.choice(len(test), size=args.n, replace=False)]
    texts = [str(t) for t in sample["clean_text"]]
    y_true = sample["label_id"].tolist()

    # ---- 1. SHAP -----------------------------------------------------------
    masker = shap.maskers.Text(r"\s")          # whitespace words as units
    explainer = shap.Explainer(f, masker, output_names=[ID2LABEL[i] for i in range(3)])
    sv = explainer(texts, batch_size=32)
    preds = f(texts).argmax(1).tolist()

    records = []
    for i, text in enumerate(texts):
        cls = preds[i]
        words = [w.strip() for w in sv.data[i]]
        attr = sv.values[i][:, cls].tolist()
        records.append({"id": str(sample.iloc[i]["id"]), "text": text,
                        "true": ID2LABEL[y_true[i]], "pred": ID2LABEL[cls],
                        "words": words, "attributions": attr})
    (OUT / "shap_values.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SHAP done on {len(records)} samples -> {OUT/'shap_values.json'}")

    # ---- 2. lexicon aggregation -------------------------------------------
    lex_rows = []
    for cls_name, entries in LEXICON.items():
        for canon, variants in entries.items():
            vset = {norm_word(v) for v in variants}
            vals = [abs(a) for r in records
                    for w, a in zip(r["words"], r["attributions"])
                    if norm_word(w) in vset]
            all_vals = [abs(a) for r in records for a in r["attributions"]]
            lex_rows.append({"class": cls_name, "word": canon,
                             "occurrences": len(vals),
                             "mean_abs_attr": float(np.mean(vals)) if vals else None,
                             "corpus_mean_abs_attr": float(np.mean(all_vals))})
    (OUT / "lexicon_attribution.json").write_text(
        json.dumps(lex_rows, indent=2), encoding="utf-8")
    for r in lex_rows:
        print(f"  {r['class']:11s} {r['word']:7s} n={r['occurrences']:3d} "
              f"mean|attr|={r['mean_abs_attr'] if r['mean_abs_attr'] is None else round(r['mean_abs_attr'],4)}")

    # ---- 3. faithfulness ---------------------------------------------------
    def mask_top(text, words, attr, k=5, random=False):
        idx = (rng.choice(len(words), size=min(k, len(words)), replace=False)
               if random else np.argsort(attr)[::-1][:k])
        keep = [w for j, w in enumerate(words) if j not in set(idx.tolist())]
        return " ".join(keep) if keep else "…"

    masked_top = [mask_top(r["text"], r["words"], r["attributions"]) for r in records]
    masked_rnd = [mask_top(r["text"], r["words"], r["attributions"], random=True)
                  for r in records]
    f1_orig = evaluate(y_true, preds)["macro_f1"]
    f1_top = evaluate(y_true, f(masked_top).argmax(1).tolist())["macro_f1"]
    f1_rnd = evaluate(y_true, f(masked_rnd).argmax(1).tolist())["macro_f1"]
    faith = {"n": len(records), "macro_f1_original": round(f1_orig, 4),
             "macro_f1_top5_masked": round(f1_top, 4),
             "macro_f1_random5_masked": round(f1_rnd, 4),
             "drop_top5": round(f1_orig - f1_top, 4),
             "drop_random5": round(f1_orig - f1_rnd, 4)}
    (OUT / "faithfulness.json").write_text(json.dumps(faith, indent=2), encoding="utf-8")
    print("faithfulness:", faith)

    # ---- 4. minimal pairs --------------------------------------------------
    pairs = []
    for a, b in MINIMAL_PAIRS:
        pa, pb = f([a])[0], f([b])[0]
        pairs.append({"a": a, "pred_a": ID2LABEL[int(pa.argmax())], "probs_a": pa.round(3).tolist(),
                      "b": b, "pred_b": ID2LABEL[int(pb.argmax())], "probs_b": pb.round(3).tolist()})
        print(f"  '{a}' -> {ID2LABEL[int(pa.argmax())]} | '{b}' -> {ID2LABEL[int(pb.argmax())]}")
    (OUT / "minimal_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figures (grayscale, from the numbers just computed) ---------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300})

    rows = [r for r in lex_rows if r["occurrences"] > 0]
    if rows:
        fig, ax = plt.subplots(figsize=(5.4, 2.6))
        xs = np.arange(len(rows))
        ax.bar(xs, [r["mean_abs_attr"] for r in rows], width=0.6,
               facecolor="0.55", edgecolor="black", lw=0.8)
        ax.axhline(rows[0]["corpus_mean_abs_attr"], color="black", ls=":", lw=1)
        ax.text(len(rows) - 0.5, rows[0]["corpus_mean_abs_attr"] * 1.05,
                "corpus mean", fontsize=7, ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['word']}\n({r['class'][:5]}., n={r['occurrences']})"
                            for r in rows], fontsize=7.5)
        ax.set_ylabel("mean |SHAP attribution|")
        ax.set_axisbelow(True); ax.grid(axis="x", visible=False)
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"lexicon_attribution.{ext}", bbox_inches="tight")
        plt.close(fig)
        print("wrote", FIGS / "lexicon_attribution.png/.pdf")


if __name__ == "__main__":
    main()
