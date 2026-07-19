# Archived: pre-schema.py pipeline (unused)

Moved here 2026-07-19, not deleted, so the history stays in git.

These nine files are an early, superseded preprocessing/modelling layer that
predates `src/common/schema.py`. Nothing in the active pipeline (`src/preprocess/
{validate_schema,quarantine,clean_text,language_id,make_splits,tokenize_cache}.py`,
`src/train/*.py`, `src/serve/api.py`, `src/explain/*.py`) imports any of them —
verified by grep before archiving.

Do not resurrect without rewriting: `dataset_builder.py` expects a two-row-header
CSV and a different raw filename, and its `LABEL_MAP` (`positive=0, negative=1,
neutral=2`) conflicts with the canonical `schema.py` mapping (`negative=0,
neutral=1, positive=2`) used everywhere else — importing these files into a
live run would silently scramble label ids.

- `dataset_builder.py`, `cleaner.py`, `sinhala_normalizer.py`, `lid_tagger.py` —
  old preprocessing chain (2-row header, `singlish_mixed_sentiment_labeled.csv`)
- `baseline_tfidf.py`, `baseline_bilstm.py`, `baseline_mbert.py`,
  `baseline_xlmr.py`, `common.py` — old model layer (`data/processed/{split}.csv`,
  `cleaned_text`/`label` columns); superseded by `src/train/run_*.py` +
  `src/train/data_io.py`, which are what actually produced the v1 results.
