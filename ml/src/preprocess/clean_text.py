"""Pipeline step 3: regenerate clean_text + text_length from a documented rule.

Input : data/interim/clean.csv          (step 2 output; never modified)
Output: data/interim/cleaned.csv        (same rows/columns, clean_text and
                                         text_length filled)

Cleaning rule (cite verbatim in thesis methodology):
  1. HTML entities unescaped (&amp; -> &)
  2. Unicode NFC normalization (canonical composition of Sinhala vowel signs)
  3. URLs replaced by a space
  4. @mentions replaced by a space
  5. Zero-width characters removed, EXCEPT U+200D ZWJ — required for Sinhala
     conjunct forms (yansaya / rakaransaya)
  6. All whitespace runs (newlines, tabs, repeats) collapsed to single space
  7. Leading/trailing whitespace stripped
  Case, punctuation and emoji are PRESERVED — they carry sentiment signal
  (XLM-R is case-sensitive; emoji-only rows are strong signal).

text_length = character count of clean_text.

Run from ml/:
    python -m src.preprocess.clean_text
"""

import hashlib
import html
import re
import subprocess
import unicodedata
from datetime import date

import pandas as pd

from src.common.schema import INTERIM_DIR, TEXT_COL

IN_CSV = INTERIM_DIR / "clean.csv"
OUT_CSV = INTERIM_DIR / "cleaned.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
# keep U+200D (ZWJ): meaningful in Sinhala script
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200e\u200f\ufeff]")
WS_RE = re.compile(r"\s+")


def clean(text):
    if pd.isna(text):
        return ""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFC", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = ZERO_WIDTH_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


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


def main():
    df = pd.read_csv(IN_CSV, encoding="utf-8", dtype=str)
    df[TEXT_COL] = df["text"].map(clean)
    df["text_length"] = df[TEXT_COL].str.len()

    n_empty = int((df[TEXT_COL] == "").sum())
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

    print(f"rows:              {len(df)}  -> {OUT_CSV}")
    print(f"empty clean_text:  {n_empty} (kept, not dropped — review if > 0)")
    print(f"mean text_length:  {df['text_length'].astype(int).mean():.1f}")
    print(f"max  text_length:  {df['text_length'].astype(int).max()}")

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/interim/cleaned.csv")
    print(f"- source:   data/interim/clean.csv (sha256: {sha256_of(IN_CSV)[:16]}...)")
    print(f"- script:   src/preprocess/clean_text.py @ git commit {git_head()}")
    print(f"- output:   cleaned.csv {len(df)} rows (sha256: {sha256_of(OUT_CSV)[:16]}...)")
    print(f"- date:     {date.today().isoformat()}")
    print("- rationale: regenerate clean_text + text_length from documented "
          "7-step rule (URLs/mentions removed, NFC, ZWJ preserved, emoji kept)")


if __name__ == "__main__":
    main()