"""LIME cross-check (secondary explainability method, per objective 3).

Explains the minimal-pair sentences plus a sample of test comments with
LIME, and measures agreement with occlusion attributions (top-3 word
overlap). Runs on CPU using the local P1 checkpoint.

Run from ml/:
    python -m src.explain.run_lime
"""

import json

import numpy as np
from lime.lime_text import LimeTextExplainer

from src.common.schema import ID2LABEL, RESULTS_DIR, SEED
from src.serve.api import probs_for  # loads the model at import time
from src.train.data_io import load_split_frames

OUT = RESULTS_DIR / "explain"
N_TEST_SAMPLES = 10
NUM_SAMPLES = 500   # LIME perturbations per sentence (CPU budget)

PAIR_SENTENCES = [
    "film eka hodai", "film eka hodai ne",
    "me phone eka harima hodai", "me phone eka hodai",
    "service eka godak hondai", "service eka hondai",
    "mama meka kamathi", "mama meka kamathi ne",
    "ela machan supiri", "epa machan meka ganna",
]


def predict_proba(texts):
    return np.asarray(probs_for(list(texts)))


def occlusion_weights(text):
    words = text.split()
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) or "…"
                         for i in range(len(words))]
    p = predict_proba(variants)
    cls = int(p[0].argmax())
    return words, [float(p[0][cls] - p[i + 1][cls]) for i in range(len(words))], cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    explainer = LimeTextExplainer(
        class_names=[ID2LABEL[i] for i in range(3)],
        split_expression=r"\s+", bow=False, random_state=SEED)

    test = load_split_frames()["test"]
    multi = test[test["clean_text"].astype(str).str.split().str.len().between(3, 25)]
    rng = np.random.default_rng(SEED)
    sampled = [str(t) for t in
               multi.iloc[rng.choice(len(multi), N_TEST_SAMPLES, replace=False)]
               ["clean_text"]]

    records, overlaps = [], []
    for i, text in enumerate(PAIR_SENTENCES + sampled):
        words, occ, cls = occlusion_weights(text)
        exp = explainer.explain_instance(
            text, predict_proba, labels=(cls,),
            num_features=len(words), num_samples=NUM_SAMPLES)
        lime_w = dict(exp.as_list(label=cls))
        lime_scores = [float(lime_w.get(w, 0.0)) for w in words]

        k = min(3, len(words))
        top_occ = set(np.argsort(occ)[::-1][:k].tolist())
        top_lime = set(np.argsort(lime_scores)[::-1][:k].tolist())
        overlap = len(top_occ & top_lime) / k
        overlaps.append(overlap)

        records.append({"text": text, "pred": ID2LABEL[cls],
                        "words": words,
                        "lime": [round(s, 4) for s in lime_scores],
                        "occlusion": [round(s, 4) for s in occ],
                        "top3_overlap": round(overlap, 2)})
        print(f"[{i + 1}/{len(PAIR_SENTENCES) + len(sampled)}] "
              f"{text[:44]!r} pred={ID2LABEL[cls]} top-3 overlap={overlap:.2f}")

    summary = {"n_sentences": len(records),
               "lime_num_samples": NUM_SAMPLES,
               "mean_top3_overlap_lime_vs_occlusion": round(float(np.mean(overlaps)), 3)}
    (OUT / "lime_results.json").write_text(
        json.dumps({"summary": summary, "records": records},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("summary:", summary)
    print("->", OUT / "lime_results.json")


if __name__ == "__main__":
    main()
