"""Dedup-robustness check: scores on the test rows whose clean_text does NOT
appear in the training split (removes the duplicate-leakage effect).

Evaluates P1 (from the local checkpoint, CPU) and B1 TF-IDF (retrained in
seconds) on the full test set vs the deduplicated subset.

Run from ml/:
    python -m src.eval.eval_dedup
"""

import json

from src.common.schema import LABEL2ID, RESULTS_DIR
from src.train.data_io import load_split_frames
from src.train.results_io import evaluate


def main():
    splits = load_split_frames()
    train, test = splits["train"], splits["test"]
    train_texts = set(train["clean_text"].astype(str))
    dedup_mask = ~test["clean_text"].astype(str).isin(train_texts)
    print(f"test rows: {len(test)}; not seen in training: {int(dedup_mask.sum())}")

    y_all = test["label_id"].tolist()

    # ---- B1 TF-IDF (retrain, seconds) -------------------------------------
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000,
                         sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100000,
                         sublinear_tf=True)
    Xtr = hstack([wv.fit_transform(train["clean_text"].fillna("")),
                  cv.fit_transform(train["clean_text"].fillna(""))])
    Xte = hstack([wv.transform(test["clean_text"].fillna("")),
                  cv.transform(test["clean_text"].fillna(""))])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             random_state=42).fit(Xtr, train["label_id"].tolist())
    preds_b1 = clf.predict(Xte).tolist()

    # ---- P1 from local checkpoint (CPU) -----------------------------------
    from src.serve.api import probs_for  # loads model at import (few minutes)
    texts = [str(t) for t in test["clean_text"]]
    preds_p1 = []
    for s in range(0, len(texts), 64):
        preds_p1.extend(probs_for(texts[s:s + 64]).argmax(1).tolist())
        if s % 512 == 0:
            print(f"  P1 inference {s}/{len(texts)}")

    out = {}
    for name, preds in (("tfidf_lr", preds_b1), ("lora_scl_lid", preds_p1)):
        full = evaluate(y_all, preds)["macro_f1"]
        yd = [y for y, m in zip(y_all, dedup_mask) if m]
        pd_ = [p for p, m in zip(preds, dedup_mask) if m]
        dedup = evaluate(yd, pd_)["macro_f1"]
        out[name] = {"macro_f1_full_test": round(full, 4),
                     "macro_f1_dedup_test": round(dedup, 4),
                     "n_dedup": len(yd)}
        print(f"{name}: full {full:.4f} -> dedup {dedup:.4f}")

    path = RESULTS_DIR / "dedup_eval.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
