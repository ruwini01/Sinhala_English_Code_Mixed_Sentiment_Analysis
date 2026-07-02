# data/raw/ — READ-ONLY

Holds the single merged raw dataset:

- `singlish_mixed_sentiment_complete.csv` (~10,409 rows, 13 columns)

This file is **immutable**: no script may write to this directory. It is
gitignored (kept local until the dataset is cleared for publication); its
sha256 is recorded in `ml/DATA.md` by `src/preprocess/validate_schema.py`
so any copy can be verified against the one used in the thesis.

If this file is missing, restore it from backup (Google Drive-2020ICT31) —
it cannot be regenerated.
