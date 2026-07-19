"""QA audit of the hand-curated language_type column in the raw dataset.

The raw file's language_type (sinhala / english / singlish / code-mixed) was
not annotated by the thesis author, so it must be verified before any
language-based analysis (RQ2, Tables 3.3/3.5/4.3/4.4) relies on it.

Method: every label implies script-level facts that can be checked
mechanically for all rows:

  label       must hold
  ----------  ------------------------------------------------------------
  sinhala     contains Sinhala script; little/no Latin word content
  english     Latin only; few/no romanized-Sinhala lexicon words
  singlish    Latin only; romanized-Sinhala lexicon words present
  code-mixed  Sinhala script AND Latin words together (script-level mixing)

Rows that break these invariants are flagged with a severity:
  HARD = script evidence contradicts the label outright (near-certain error)
  SOFT = boundary case (english vs singlish vs code-mixed on Latin-only
         text depends on lexicon coverage — review a sample, not all)

Outputs (results/tables/):
  lid_audit_full.csv    every row + evidence + verdict
  lid_audit_queue.csv   flagged rows only, HARD first — the review queue
and a printed crosstab of manual label vs rule-based expectation.

Run from ml/:
    python -m src.preprocess.audit_language_type
"""

import re

import pandas as pd

from src.common.schema import ID_COL, LABEL_COL, ML_ROOT, RAW_CSV, SPLITS_DIR

LEXICON_TXT = ML_ROOT / "data" / "lexicon" / "romanized_sinhala_extended.txt"
OUT_DIR = ML_ROOT / "results" / "tables"

SINHALA_RE = re.compile(r"[඀-෿]")
LATIN_WORD_RE = re.compile(r"[a-zA-Z]{2,}")

# Latin-only text: lexicon-hit share >= this => romanized Sinhala is present
SINGLISH_MIN_HIT_SHARE = 0.15
# Annotator taxonomy (verified by inspection): ANY Latin word alongside
# Sinhala script counts as code-mixed ("factory එකක්" is code-mixed)
SINHALA_MAX_LATIN_WORDS = 0

MANUAL_COL = "language_type"  # in the RAW file this is the hand-curated column


def expected_label(text, lexicon):
    """Rule-based expectation + evidence for one comment."""
    t = str(text)
    has_sinhala = bool(SINHALA_RE.search(t))
    latin_words = [w.lower() for w in LATIN_WORD_RE.findall(t)]
    hits = [w for w in latin_words if w in lexicon]
    hit_share = len(hits) / len(latin_words) if latin_words else 0.0

    if has_sinhala and len(latin_words) > SINHALA_MAX_LATIN_WORDS:
        exp = "code-mixed"
    elif has_sinhala:
        exp = "sinhala"
    elif latin_words and hit_share >= SINGLISH_MIN_HIT_SHARE:
        exp = "singlish"
    elif latin_words:
        exp = "english"
    else:
        exp = "unknown"
    return exp, has_sinhala, len(latin_words), len(hits), round(hit_share, 3)


def verdict(manual, expected):
    """OK / HARD / SOFT for a (manual label, rule expectation) pair."""
    if manual == expected:
        return "OK"
    hard = {
        # script evidence contradicts the label outright
        ("sinhala", "english"), ("sinhala", "singlish"), ("sinhala", "unknown"),
        ("english", "sinhala"), ("english", "code-mixed"),
        ("singlish", "sinhala"), ("singlish", "code-mixed"),
        ("code-mixed", "english"), ("code-mixed", "singlish"),
        ("code-mixed", "sinhala"), ("code-mixed", "unknown"),
    }
    return "HARD" if (manual, expected) in hard else "SOFT"


def main():
    lexicon = set(LEXICON_TXT.read_text(encoding="utf-8").split())
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str)

    split_of = {}
    for split in ("train", "val", "test"):
        f = SPLITS_DIR / f"{split}_ids.txt"
        if f.exists():
            for rid in f.read_text(encoding="utf-8").split():
                split_of[rid] = split
    df["split"] = df[ID_COL].map(split_of).fillna("unsplit")

    ev = df["text"].map(lambda t: expected_label(t, lexicon))
    df["expected"] = ev.map(lambda x: x[0])
    df["has_sinhala_script"] = ev.map(lambda x: x[1])
    df["latin_words"] = ev.map(lambda x: x[2])
    df["lexicon_hits"] = ev.map(lambda x: x[3])
    df["lexicon_hit_share"] = ev.map(lambda x: x[4])
    df["manual"] = df[MANUAL_COL].str.strip().str.lower()
    df["verdict"] = [verdict(m, e) for m, e in zip(df["manual"], df["expected"])]

    n = len(df)
    ok = (df["verdict"] == "OK").sum()
    hard = (df["verdict"] == "HARD").sum()
    soft = (df["verdict"] == "SOFT").sum()
    print(f"rows: {n}   agree: {ok} ({ok/n:.1%})   "
          f"HARD violations: {hard} ({hard/n:.1%})   SOFT: {soft} ({soft/n:.1%})")

    print("\nmanual label (rows) x rule expectation (cols):")
    print(pd.crosstab(df["manual"], df["expected"]).to_string())

    print("\nHARD violations by (manual -> expected):")
    hq = df[df["verdict"] == "HARD"]
    print(hq.groupby(["manual", "expected"]).size().sort_values(ascending=False).to_string())

    print("\nHARD violations by split:")
    print(hq["split"].value_counts().to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [ID_COL, "text", LABEL_COL, "split", "manual", "expected", "verdict",
            "has_sinhala_script", "latin_words", "lexicon_hits", "lexicon_hit_share"]
    df[cols].to_csv(OUT_DIR / "lid_audit_full.csv", index=False, encoding="utf-8")

    queue = df.loc[df["verdict"] != "OK", cols].sort_values(
        ["verdict", "manual"], ascending=[True, True])  # HARD sorts before SOFT
    queue.to_csv(OUT_DIR / "lid_audit_queue.csv", index=False, encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'lid_audit_full.csv'}  ({n} rows)")
    print(f"wrote {OUT_DIR / 'lid_audit_queue.csv'}  ({len(queue)} rows, HARD first)")


if __name__ == "__main__":
    main()
