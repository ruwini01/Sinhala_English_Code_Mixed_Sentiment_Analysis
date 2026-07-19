"""Pipeline step 4: regenerate language_type deterministically.

Input : data/interim/cleaned.csv       (step 3 output; never modified)
Output: data/interim/lid.csv           (language_type overwritten;
                                        language_type_manual preserves the
                                        raw file's hand-curated labels;
                                        label_source column added)

The raw file's language_type (v2: sinhala / english / singlish / code-mixed)
is hand-curated and distinguishes romanized Sinhala ("singlish") from
script-mixed text ("code-mixed") — something this script's lexicon rule
cannot do. It is kept verbatim in language_type_manual for thesis figures
and per-language evaluation; the deterministic column below stays the
reproducible pipeline output.

Rule (deterministic, cite in thesis methodology):
  Token level (whitespace tokens of clean_text):
    - "sin"     if the token contains any Sinhala-script char (U+0D80-U+0DFF)
    - "sin"     if the token is Latin-script but appears in the romanized
                Sinhala lexicon below (Singlish written in Latin letters)
    - "eng"     if it contains any A-Z/a-z and is not in the lexicon
    - "other"   otherwise (digits, punctuation, emoji)
  Row level (language_type):
    - "mixed"    if the row has both sin and eng tokens
    - "sinhala"  if only sin
    - "english"  if only eng
    - "unknown"  if neither (emoji-only rows — kept, strong sentiment signal)

label_source = "deterministic" for every row (this script is the labeler).

Run from ml/:
    python -m src.preprocess.language_id
"""

import hashlib
import re
import subprocess
from datetime import date

import pandas as pd

from src.common.schema import INTERIM_DIR, LID_COL, MANUAL_LID_COL, TEXT_COL

IN_CSV = INTERIM_DIR / "cleaned.csv"
OUT_CSV = INTERIM_DIR / "lid.csv"

SINHALA_CHAR_RE = re.compile(r"[\u0D80-\u0DFF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

# Romanized Sinhala lexicon: high-frequency Singlish function/content words.
# Latin-script tokens matching these (case-insensitive) count as Sinhala.
# Kept deliberately conservative — words that are also common English words
# (e.g. "me", "oya" vs "oy") must NOT be added unless unambiguous.
ROMANIZED_SINHALA = {
    # intensifiers / negations (thesis claim 2 lexicon)
    "harima", "godak", "hari", "ne", "nehe", "naa", "epa", "nathi", "bea", "ba",
    # common verbs / particles
    "ekak", "eka", "wage", "wagey", "kiyala", "kiyanne", "puluwan", "puluwang",
    "thama", "thamai", "witharai", "vitharai", "nam", "nang", "kohomada",
    "mokada", "monawada", "innawa", "yanawa", "enawa", "karanawa", "karanna",
    "ganna", "denna", "balanna", "dannawa", "danne", "hithanawa", "hithanne",
    # pronouns / people
    "mama", "mata", "mage", "oya", "oyage", "eyala", "apita", "api", "ape",
    "machan", "malli", "aiya", "akka", "nangi",
    # sentiment-heavy slang
    "hodai", "hondai", "honda", "lassanai", "lassana", "patta", "ela",
    "elakiri", "niyamai", "maru", "shok", "aiyo", "ayyo", "chi", "chik",
    "kunuharupa", "boru", "pissu", "modaya", "gon", "haduwa",
}


def token_lang(token):
    if SINHALA_CHAR_RE.search(token):
        return "sin"
    if LATIN_CHAR_RE.search(token):
        stripped = re.sub(r"[^A-Za-z]", "", token).lower()
        return "sin" if stripped in ROMANIZED_SINHALA else "eng"
    return "other"


def row_lang(text):
    if pd.isna(text) or not str(text).strip():
        return "unknown"
    tags = {token_lang(t) for t in str(text).split()}
    has_sin, has_eng = "sin" in tags, "eng" in tags
    if has_sin and has_eng:
        return "mixed"
    if has_sin:
        return "sinhala"
    if has_eng:
        return "english"
    return "unknown"


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

    df[MANUAL_LID_COL] = df[LID_COL]  # hand-curated labels, kept verbatim
    manual = df[MANUAL_LID_COL].str.strip().str.lower()
    df[LID_COL] = df[TEXT_COL].map(row_lang)
    df["label_source"] = "deterministic"

    agree = int((manual == df[LID_COL]).sum())
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

    print(f"rows: {len(df)}  -> {OUT_CSV}")
    print("\ndeterministic language_type distribution:")
    print(df[LID_COL].value_counts().to_string())
    print(f"\nmanual language_type distribution (preserved in {MANUAL_LID_COL}):")
    print(manual.value_counts(dropna=False).to_string())
    print(f"\nmanual/deterministic agreement: {agree}/{len(df)} "
          f"({100 * agree / len(df):.1f}%) — taxonomies differ, low is expected")

    print("\n--- paste into ml/DATA.md under 'Derived artifacts' ---")
    print("## data/interim/lid.csv")
    print(f"- source:   data/interim/cleaned.csv (sha256: {sha256_of(IN_CSV)[:16]}...)")
    print(f"- script:   src/preprocess/language_id.py @ git commit {git_head()}")
    print(f"- output:   lid.csv {len(df)} rows (sha256: {sha256_of(OUT_CSV)[:16]}...)")
    print(f"- date:     {date.today().isoformat()}")
    print("- rationale: language_type regenerated by deterministic script+lexicon "
          "rule; label_source=deterministic added")


if __name__ == "__main__":
    main()