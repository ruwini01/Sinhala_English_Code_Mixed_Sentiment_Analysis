# DATA.md — Provenance Ledger

Every file under `data/interim/` or `data/processed/` gets an entry here.
Cite in the thesis methodology chapter as: "reproducibility: see DATA.md, commit `<sha>`."

Entry format:

```
## <output file path>
- source:   <input file path> (sha256: <hash>)
- script:   <script path> @ git commit <sha>
- output:   <output file path> (sha256: <hash>)
- date:     YYYY-MM-DD
- rationale: <one line>
```

---

## Raw input

## data/raw/singlish_mixed_sentiment_complete.csv (v3 — current, LID-audit corrections applied 2026-07-19)
- source:   v2.1 export + 639 language_type corrections from the manual review
            of all 1,075 rows flagged by src/preprocess/audit_language_type.py
            (decisions logged in results/tables/lid_review_filled.xlsx and
            lid_corrections.csv; sentiment labels untouched)
- sha256:   5ca2952ebf1087dbc0704beb4c53306bf3e30ab201ff72a60171565a932ba221
- rows:     10,160 total; 10,160 3-class-ready; 0 missing label; 0 non-standard label
- schema:   11 columns — id, text, sentiment_label, source_platform, source_url,
            language_type, clean_text, text_length, domain, content_type, emotion
- notes:    language_type is HAND-CURATED on a 4-way taxonomy, audited and
            corrected 2026-07-19 — singlish 6071 (fully romanized Sinhala),
            code-mixed 1896 (Sinhala script + English), sinhala 1266,
            english 908 — plus 11 "other" and 8 "unknown" stragglers;
            script-level consistency after correction: 95.7% (6 deliberate
            keeps remain as HARD flags — transliterated loanwords);
            preserved verbatim as language_type_manual by pipeline step 4,
            which still writes its own deterministic language_type for the
            reproducibility chain (the two taxonomies differ by design);
            clean_text and text_length come PRE-FILLED (unlike v1); pipeline
            step 3 still regenerates both from `text` under the documented
            7-step rule — verified byte-identical output on this file;
            file carries a UTF-8 BOM (read with encoding="utf-8-sig");
            source_platform has case drift (youtube 3888/YouTube 512,
            facebook 3602/Facebook 145) and 2 junk values ("singlish",
            "code-mixing") — handled downstream by quarantine.py's
            normalize_platform(), raw never edited
- verified: 2026-07-19 by src/preprocess/validate_schema.py
- status:   IMMUTABLE. Never edited. All artifacts derive from it.
- history:  supersedes v2.1 (sha256 48ef313f..., the pre-audit language_type
            values) and the v2.0 export of the same day (sha256 5ed7cc85...);
            text/sentiment/ids identical across v2.0→v3, only language_type
            (and 1-2 platform cells) changed. The v1 raw file (10,539 rows,
            sha256 0b513cff...) and all v1 derived artifacts are preserved
            below under "v1 (historical)" and remain reproducible from git
            history at commit c5731b1^ onward.

### Collection provenance (thesis §3.2, verbatim)
- YouTube comments — python scraper (`src/ingest/youtube_collector.py`)
- TikTok comments — Apify TikTok comment scraper (JSON export)
- Play Store reviews — python scraper
- Facebook — manual collection

Per-platform raw files were not preserved separately; `source_platform` is the
authoritative record of origin and is preserved byte-exact.

---

## Derived artifacts

### v3 (current, 2026-07-19 — LID-audit corrections + lexicon-wired LID + CMI + group-aware splits)

## data/interim/clean.csv + data/interim/quarantine.csv
- source:   data/raw/singlish_mixed_sentiment_complete.csv (sha256: 5ca2952ebf1087db...)
- script:   src/preprocess/quarantine.py @ git commit b4a79d3
- output:   clean.csv 10160 rows (sha256: 0d197fe1bd80209c...); quarantine.csv 0 rows (sha256: 9f7dd093cd3f7591...)
- date:     2026-07-19
- rationale: separate 3-class-ready rows; quarantined labels kept verbatim, never remapped

## data/interim/cleaned.csv
- source:   data/interim/clean.csv (sha256: 0d197fe1bd80209c...)
- script:   src/preprocess/clean_text.py @ git commit b4a79d3
- output:   cleaned.csv 10160 rows (sha256: 0d197fe1bd80209c... — byte-identical
            to clean.csv: the raw clean_text already matches the 7-step rule)
- date:     2026-07-19
- rationale: regenerate clean_text + text_length from documented 7-step rule

## data/interim/lid.csv
- source:   data/interim/cleaned.csv (sha256: 0d197fe1bd80209c...)
- script:   src/preprocess/language_id.py @ git commit b4a79d3 (mined 1,806-word
            romanized lexicon wired in, minus 2 English-ambiguous entries; adds cmi column)
- output:   lid.csv 10160 rows (sha256: 5d62e97d0bc952c9...)
- date:     2026-07-19
- rationale: deterministic language_type (mixed 7318 / sinhala 1600 / english 1223 /
            unknown 19) + language_type_manual preserved + CMI (Das & Gambäck;
            mean 23.2 all rows, 32.2 mixed-only)

## data/processed/splits/{train,val,test}_ids.txt + excluded_conflict_ids.txt
- source:   data/interim/lid.csv (sha256: 5d62e97d0bc952c9...)
- script:   src/preprocess/make_splits.py @ git commit b4a79d3 (GROUP-AWARE)
- output:   train 7096, val 1524, test 1525; 15 conflicting-label duplicate rows
            (6 groups) excluded (committed to git; LOCKED, never re-split)
- date:     2026-07-19
- rationale: stratified 70/15/15 over normalized-text groups, seed=42; duplicate
            texts share a split; ZERO normalized-text overlap verified in-script

## data/processed/tokenized/{train,val,test}.pt
- source:   data/interim/lid.csv (sha256: 5d62e97d0bc952c9...) + splits/*_ids.txt
- script:   src/preprocess/tokenize_cache.py @ git commit b4a79d3
- output:   train.pt 7096 rows (sha256: 87959fe86607a20b...); val.pt 1524 rows
            (sha256: f3815f1f0636780e...); test.pt 1525 rows (sha256: 2a8d0064420e0f27...)
- date:     2026-07-19
- rationale: one-time xlm-roberta-base tokenization (max_len=128) with per-token
            LID tags (lexicon-aware); all experiments load this cache

### v2.1 (historical — superseded by v3 the same day; pre-audit language_type)

## data/interim/clean.csv + data/interim/quarantine.csv
- source:   data/raw/singlish_mixed_sentiment_complete.csv (sha256: 48ef313fb7174118...)
- script:   src/preprocess/quarantine.py @ git commit a833574
- output:   clean.csv 10160 rows (sha256: 5735944335949eaa...); quarantine.csv 0 rows (sha256: 9f7dd093cd3f7591...)
- date:     2026-07-19
- rationale: separate 3-class-ready rows (all 10,160 rows valid —
            quarantine.csv is empty); quarantined labels kept verbatim, never
            remapped; source_platform normalized (case drift + junk values ->
            youtube/facebook/tiktok/google_play/unknown) via normalize_platform()

## data/interim/cleaned.csv
- source:   data/interim/clean.csv (sha256: 5735944335949eaa...)
- script:   src/preprocess/clean_text.py @ git commit a833574
- output:   cleaned.csv 10160 rows (sha256: 5735944335949eaa... — byte-identical
            to clean.csv: the raw file's supplied clean_text already matched
            the documented 7-step rule)
- date:     2026-07-19
- rationale: regenerate clean_text + text_length from documented 7-step rule (URLs/mentions removed, NFC, ZWJ preserved, emoji kept)

## data/interim/lid.csv
- source:   data/interim/cleaned.csv (sha256: 5735944335949eaa...)
- script:   src/preprocess/language_id.py @ git commit a833574 (+ uncommitted
            manual-column preservation)
- output:   lid.csv 10160 rows (sha256: 45ef7b52f09be62e...)
- date:     2026-07-19
- rationale: language_type regenerated by deterministic script+lexicon rule
            (mixed 5883 / english 2954 / sinhala 1304 / unknown 19);
            hand-curated raw labels preserved verbatim in language_type_manual
            (singlish 6096 / code-mixed 1679 / sinhala 1456 / english 910 /
            other 11 / unknown 8) for thesis figures and per-language eval;
            label_source=deterministic added

## data/processed/splits/{train,val,test}_ids.txt
- source:   data/interim/lid.csv (sha256: 45ef7b52f09be62e...)
- script:   src/preprocess/make_splits.py @ git commit c5731b1
- output:   train 7112, val 1524, test 1524 (untracked as of 2026-07-19 —
            commit to git to LOCK; never re-split once committed)
- date:     2026-07-19
- note:     generated against the v2.0 lid.csv (sha256 4f14b545...) and NOT
            re-split after the language_type recuration: make_splits reads
            only id + sentiment_label, and both are byte-identical between
            v2.0 and v2.1, so the id lists are unchanged and remain valid
            (verified 2026-07-19 by loading all three splits against the
            v2.1 lid.csv via src/train/data_io.py)
- rationale: stratified 70/15/15 on sentiment_label, seed=42

## data/processed/tokenized/{train,val,test}.pt
- source:   data/interim/lid.csv (sha256: 45ef7b52f09be62e...) + splits/*_ids.txt
- script:   src/preprocess/tokenize_cache.py @ git commit c5731b1
- output:   train.pt 7112 rows; val.pt 1524 rows; test.pt 1524 rows
- date:     2026-07-19
- note:     generated against the v2.0 lid.csv and NOT re-tokenized after the
            language_type recuration: tokenize_cache reads only clean_text,
            sentiment_label and id (per-token LID tags are computed from
            clean_text by the deterministic rule, not from the language_type
            column), and all three inputs are identical between v2.0 and v2.1
- rationale: one-time xlm-roberta-base tokenization (max_len=128) with per-token
            LID tags; all experiments load this cache (back up to Google Drive
            before starting Colab training — .pt files are gitignored)

### v1 (historical — superseded 2026-07-19, reproducible from git history at commit c5731b1^)

## data/interim/clean.csv + data/interim/quarantine.csv
- source:   data/raw/singlish_mixed_sentiment_complete.csv (sha256: 0b513cff6676969b...)
- script:   src/preprocess/quarantine.py @ git commit d3c7811
- output:   clean.csv 10509 rows (sha256: 321216ccd3239547...); quarantine.csv 30 rows (sha256: cfa262584c6fa901...)
- date:     2026-07-05
- rationale: separate 3-class-ready rows; quarantined labels kept verbatim, never remapped
- verified: identical sha256 on Windows 11 (local) and Linux (Colab)

## data/interim/cleaned.csv
- source:   data/interim/clean.csv (sha256: 321216ccd3239547...)
- script:   src/preprocess/clean_text.py @ git commit d3c7811
- output:   cleaned.csv 10509 rows (sha256: dd79f0eef7aaf19a...)
- date:     2026-07-05
- rationale: regenerate clean_text + text_length from documented 7-step rule (URLs/mentions removed, NFC, ZWJ preserved, emoji kept)
- verified: identical sha256 on Windows 11 (local) and Linux (Colab)

## data/interim/lid.csv
- source:   data/interim/cleaned.csv (sha256: dd79f0eef7aaf19a...)
- script:   src/preprocess/language_id.py @ git commit d3c7811
- output:   lid.csv 10509 rows (sha256: 0d91daf6bc9dbb73...)
- date:     2026-07-05
- rationale: language_type regenerated by deterministic script+lexicon rule; label_source=deterministic added
- note:     old/new agreement 19.6% is computed on raw strings; the previous
            column used a different taxonomy (singlish/code-mixed)
- verified: identical sha256 on Windows 11 (local) and Linux (Colab)

## data/processed/splits/{train,val,test}_ids.txt
- source:   data/interim/lid.csv (sha256: 0d91daf6bc9dbb73...)
- script:   src/preprocess/make_splits.py @ git commit cda59d6
- output:   train 7356, val 1576, test 1577 (superseded — replaced by v2 split above)
- date:     2026-07-05
- rationale: stratified 70/15/15 on sentiment_label, seed=42; row ids are
            line-ending independent, so the pre-fix split remains valid

## data/processed/tokenized/{train,val,test}.pt
- source:   data/interim/lid.csv (sha256: 0d91daf6bc9dbb73...) + splits/*_ids.txt
- script:   src/preprocess/tokenize_cache.py @ git commit d807e04
- output:   train.pt 7356 rows (sha256: d184b8bcef340f6f...); val.pt 1576 rows (sha256: b0c90ec8d40ebd1f...); test.pt 1577 rows (sha256: 8e6c004e5b6a9f95...)
- date:     2026-07-05
- rationale: one-time xlm-roberta-base tokenization (max_len=128) with per-token
            LID tags; all experiments load this cache (backed up to Google Drive)

