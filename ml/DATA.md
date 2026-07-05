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

## data/raw/singlish_mixed_sentiment_complete.csv
- source:   merged export of all platform collections (see Collection provenance below)
- sha256:   TODO — filled by src/preprocess/validate_schema.py
- rows:     ~10,409 (exact count logged by validate_schema.py)
- schema:   13 columns — id, text, sentiment_label, source_platform, source_url,
            language_type, text_clean, text_length, domain, content_type,
            emotion, sentiment_fine, label_source
- status:   IMMUTABLE. Never edited. All artifacts derive from it.

### Collection provenance (thesis §3.2, verbatim)
- YouTube comments — python scraper (`src/ingest/youtube_collector.py`)
- TikTok comments — Apify TikTok comment scraper (JSON export)
- Play Store reviews — python scraper
- Facebook — manual collection

Per-platform raw files were not preserved separately; `source_platform` is the
authoritative record of origin and is preserved byte-exact.

---

## Derived artifacts

(entries appended by each pipeline step, newest last)
