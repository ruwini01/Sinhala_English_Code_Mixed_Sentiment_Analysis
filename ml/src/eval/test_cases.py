"""Run the trained P1 model on a small set of sentences and print a
comparison table: expected label, predicted label, confidence, and whether
the prediction was correct, plus the sample error rate. Also samples real
gold-labelled rows from the held-out test split. Writes a LaTeX table for the
thesis at results/tables/test_cases.tex.

Two kinds of cases are reported separately:
  1. Curated illustrative sentences. These carry the author's intended label,
     not a gold annotation, and cover romanized Sinhala, code-mixing, Sinhala
     script, negation, and intensifiers. This is a qualitative demonstration,
     NOT an evaluation metric; the metric is macro-F1 on the full 1,525-row
     test set reported in Chapter 4.
  2. A random sample of real test rows with their gold labels, so the table
     also shows behaviour on held-out data the model never saw.

Run from ml/ once the v3 P1 checkpoint sits at data/checkpoints/lora_scl_lid.pt:
    python -m src.eval.test_cases
Or, on Colab right after training P1, import and call:
    from src.eval.test_cases import load_model, run, CURATED
    tok, model = load_model("lora_scl_lid.pt")
    run(CURATED, tok, model, "Curated sentences")
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.common.schema import (
    ID2LABEL,
    ID_COL,
    INTERIM_DIR,
    LABEL_COL,
    LID_TAGS,
    RESULTS_DIR,
    SPLITS_DIR,
    TEXT_COL,
)
from src.models.proposed_xlmr_lora import ProposedModel
from src.preprocess.language_id import token_lang

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
DEFAULT_CKPT = Path("data/checkpoints/lora_scl_lid.pt")

# Curated illustrative sentences (author's intended label, NOT gold).
CURATED = [
    ("akkiyoo oya dress eka lassanai", "positive"),
    ("me phone eka harima supiri", "positive"),
    ("thank you very much for the video", "positive"),
    ("ela machan, niyamai wada", "positive"),
    ("me update eka complete waste, godak narakai", "negative"),
    ("mama meka kamathi ne", "negative"),
    ("temu eken ganna epa", "negative"),
    ("api wenuwen karapu vede hodada නෑ", "negative"),
    ("mobile da pc da", "neutral"),
    ("delivery eka kawada enne", "neutral"),
    ("kohomada order eka track karanne", "neutral"),
]


def load_model(ckpt=DEFAULT_CKPT):
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            "Download the v3 P1 checkpoint from Colab/Drive to "
            "data/checkpoints/lora_scl_lid.pt, or run this on Colab where the "
            "trained model already exists. The local .pt dated 2026-07-16 is v1."
        )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ProposedModel(model_name=MODEL_NAME)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return tok, model


def predict(tok, model, texts):
    """Return [(label, confidence), ...] for a list of raw strings.

    Mirrors the tokenisation and LID-tagging in src/serve/api.py exactly."""
    words_list = [t.split() if t.split() else ["…"] for t in texts]
    enc = tok(words_list, is_split_into_words=True, truncation=True,
              max_length=MAX_LEN, padding=True, return_tensors="pt")
    lid = torch.full_like(enc["input_ids"], LID_TAGS["special"])
    for i, words in enumerate(words_list):
        tags = [LID_TAGS.get(token_lang(w), LID_TAGS["other"]) for w in words]
        for pos, wid in enumerate(enc.word_ids(batch_index=i)):
            if wid is not None:
                lid[i, pos] = tags[wid]
    with torch.no_grad():
        logits, _ = model(enc["input_ids"], enc["attention_mask"], lid)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(1)
    return [(ID2LABEL[int(preds[i])], float(probs[i, int(preds[i])]))
            for i in range(len(texts))]


def run(cases, tok, model, title):
    out = predict(tok, model, [c[0] for c in cases])
    rows, n_correct = [], 0
    for (text, expected), (pred, conf) in zip(cases, out):
        ok = pred == expected
        n_correct += ok
        rows.append({"text": text, "expected": expected, "pred": pred,
                     "conf": conf, "correct": ok})
    acc = n_correct / len(cases)
    print(f"\n{title}: {n_correct}/{len(cases)} correct "
          f"({acc:.0%}), error rate {1 - acc:.0%}")
    for r in rows:
        flag = "OK " if r["correct"] else "XX "
        print(f"  [{flag}] {r['expected']:8s} -> {r['pred']:8s} "
              f"{r['conf']:.2f}  {r['text']}")
    return rows, acc


def sample_test_rows(n=8, seed=42):
    """Random gold-labelled rows from the committed test split."""
    df = pd.read_csv(INTERIM_DIR / "lid.csv", encoding="utf-8")
    ids = set(pd.read_csv(SPLITS_DIR / "test_ids.txt", header=None)[0])
    test = df[df[ID_COL].isin(ids)].sample(n=n, random_state=seed)
    return list(zip(test[TEXT_COL].astype(str), test[LABEL_COL]))


def to_latex(rows, path, caption, label):
    def esc(s):
        return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&")
                .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))
    lines = [
        "\\begin{table}[!ht]", "    \\centering", "    \\small",
        f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
        "    \\begin{tabular}{p{6.6cm}llr c}", "        \\hline",
        "        \\textbf{Comment} & \\textbf{Expected} & \\textbf{Predicted} "
        "& \\textbf{Conf.} & \\textbf{Correct} \\\\", "        \\hline",
    ]
    for r in rows:
        mark = "\\checkmark" if r["correct"] else "\\texttimes"
        lines.append(
            f"        {esc(r['text'])} & {r['expected']} & {r['pred']} "
            f"& {r['conf']:.2f} & {mark} \\\\")
    lines += ["        \\hline", "    \\end{tabular}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--n-test", type=int, default=8)
    args = ap.parse_args()

    tok, model = load_model(args.ckpt)
    tab_dir = RESULTS_DIR / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = run(CURATED, tok, model, "Curated illustrative sentences")
    to_latex(rows, tab_dir / "test_cases_curated.tex",
             "Predictions on curated illustrative Singlish sentences "
             "(intended labels, qualitative demonstration)",
             "tab:testcases")

    gold = sample_test_rows(args.n_test)
    grows, _ = run(gold, tok, model, "Random gold test-set rows")
    to_latex(grows, tab_dir / "test_cases_testset.tex",
             "Predictions on a random sample of gold-labelled test comments",
             "tab:testcasesgold")


if __name__ == "__main__":
    main()
