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
                Sinhala lexicon (1,806 words mined from the NLPC-UOM gold
                word-level annotations, data/lexicon/romanized_sinhala_extended.txt,
                lifting romanized recall 17.1% -> 77.8%; two entries that are
                also common English words — "ban", "wanna" — are excluded)
                or in the small curated set below
    - "eng"     if it contains any A-Z/a-z and is not in the lexicon
    - "other"   otherwise (digits, punctuation, emoji)
  Row level (language_type):
    - "mixed"    if the row has both sin and eng tokens
    - "sinhala"  if only sin
    - "english"  if only eng
    - "unknown"  if neither (emoji-only rows — kept, strong sentiment signal)

Also computes cmi (Code-Mixing Index, Das & Gambäck):
    cmi = 100 * (1 - max_lang_tokens / (total_tokens - other_tokens))
    0 for monolingual or all-"other" rows; >0 measures mixing intensity.

label_source = "deterministic" for every row (this script is the labeler).

Run from ml/:
    python -m src.preprocess.language_id
"""

import hashlib
import re
import subprocess
from datetime import date

import pandas as pd

from src.common.schema import DATA_DIR, INTERIM_DIR, LID_COL, MANUAL_LID_COL, TEXT_COL

IN_CSV = INTERIM_DIR / "cleaned.csv"
OUT_CSV = INTERIM_DIR / "lid.csv"
LEXICON_TXT = DATA_DIR / "lexicon" / "romanized_sinhala_extended.txt"

# Lexicon entries that are also common English words — matching them as
# Sinhala would mistag ordinary English ("First ban the production").
AMBIGUOUS_ENGLISH = {"ban", "wanna"}

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

# Mined lexicon (NLPC-UOM gold, 1,806 words) wired into the default rule.
# Romanized-Sinhala recall on the gold set: 17.1% without it, 77.8% with it.
ROMANIZED_SINHALA |= (
    set(LEXICON_TXT.read_text(encoding="utf-8").split()) - AMBIGUOUS_ENGLISH
)


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


def row_cmi(text):
    """Code-Mixing Index (Das & Gambäck): 0 = monolingual, higher = more mixed."""
    if pd.isna(text) or not str(text).strip():
        return 0.0
    tags = [token_lang(t) for t in str(text).split()]
    lang = [t for t in tags if t != "other"]
    if not lang:
        return 0.0
    top = max(lang.count("sin"), lang.count("eng"))
    return round(100.0 * (1.0 - top / len(lang)), 2)


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
    df["cmi"] = df[TEXT_COL].map(row_cmi)
    df["label_source"] = "deterministic"

    agree = int((manual == df[LID_COL]).sum())
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

    print(f"rows: {len(df)}  -> {OUT_CSV}")
    print(f"lexicon size: {len(ROMANIZED_SINHALA)} romanized-Sinhala words")
    print("\ndeterministic language_type distribution:")
    print(df[LID_COL].value_counts().to_string())
    cmi = df["cmi"].astype(float)
    mixed_cmi = cmi[cmi > 0]
    print(f"\nCMI: mean(all) {cmi.mean():.1f}   mean(mixed only) "
          f"{mixed_cmi.mean():.1f}   mixed rows {len(mixed_cmi)} "
          f"({100 * len(mixed_cmi) / len(df):.1f}%)")
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