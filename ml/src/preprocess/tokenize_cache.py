"""Pipeline step 6: tokenize once + build per-token LID tag ids.

Input : data/interim/lid.csv + data/processed/splits/{train,val,test}_ids.txt
Output: data/processed/tokenized/{train,val,test}.pt
        each a dict: input_ids, attention_mask, lid_ids, labels, row_ids

Every experiment loads from this cache — no experiment re-tokenizes.
Per-token LID tags use the SAME word rule as language_id.py (imported, not
re-implemented), propagated to XLM-R subwords via word_ids(); special tokens
and padding get the "special" tag.

Run from ml/ (Colab or local, needs torch + transformers):
    python -m src.preprocess.tokenize_cache
"""

import hashlib
import subprocess
from datetime import date

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.common.schema import (
    ID_COL,
    INTERIM_DIR,
    LABEL2ID,
    LABEL_COL,
    LID_TAGS,
    SPLITS_DIR,
    TEXT_COL,
    TOKENIZED_DIR,
)
from src.preprocess.language_id import token_lang

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128

IN_CSV = INTERIM_DIR / "lid.csv"


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


def encode_split(df, tokenizer):
    words_list = [str(t).split() for t in df[TEXT_COL]]
    enc = tokenizer(
        words_list,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    lid_ids = torch.full_like(enc["input_ids"], LID_TAGS["special"])
    n_truncated = 0
    for i, words in enumerate(words_list):
        word_tags = [LID_TAGS.get(token_lang(w), LID_TAGS["other"]) for w in words]
        word_ids = enc.word_ids(batch_index=i)
        seen_words = set()
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                lid_ids[i, pos] = word_tags[wid]
                seen_words.add(wid)
        if len(seen_words) < len(words):
            n_truncated += 1
    labels = torch.tensor(df[LABEL_COL].map(LABEL2ID).tolist(), dtype=torch.long)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "lid_ids": lid_ids,
        "labels": labels,
        "row_ids": df[ID_COL].tolist(),
    }, n_truncated


def main():
    df = pd.read_csv(IN_CSV, encoding="utf-8", dtype=str).set_index(ID_COL, drop=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    out_shas = []
    for name in ("train", "val", "test"):
        ids = (SPLITS_DIR / f"{name}_ids.txt").read_text(encoding="utf-8").split()
        part = df.loc[ids]
        assert len(part) == len(ids), f"{name}: id list does not match lid.csv"
        data, n_trunc = encode_split(part, tokenizer)
        out_path = TOKENIZED_DIR / f"{name}.pt"
        torch.save(data, out_path)
        out_shas.append(f"{name}.pt {len(ids)} rows (sha256: {sha256_of(out_path)[:16]}...)")
        print(f"{name:5s}: {len(ids):5d} rows, {n_trunc} truncated at {MAX_LEN} -> {out_path}")

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/processed/tokenized/{train,val,test}.pt")
    print(f"- source:   data/interim/lid.csv (sha256: {sha256_of(IN_CSV)[:16]}...) "
          f"+ splits/*_ids.txt")
    print(f"- script:   src/preprocess/tokenize_cache.py @ git commit {git_head()}")
    print(f"- output:   {'; '.join(out_shas)}")
    print(f"- date:     {date.today().isoformat()}")
    print(f"- rationale: one-time {MODEL_NAME} tokenization (max_len={MAX_LEN}) "
          "with per-token LID tags; all experiments load this cache")


if __name__ == "__main__":
    main()