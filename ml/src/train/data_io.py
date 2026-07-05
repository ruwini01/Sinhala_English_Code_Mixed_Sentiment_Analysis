"""Load the locked splits. Single implementation for all experiments."""

import pandas as pd

from src.common.schema import ID_COL, INTERIM_DIR, LABEL2ID, LABEL_COL, SPLITS_DIR

LID_CSV = INTERIM_DIR / "lid.csv"


def load_split_frames():
    """Returns {train, val, test} DataFrames from lid.csv per the locked id lists."""
    df = pd.read_csv(LID_CSV, encoding="utf-8", dtype=str).set_index(ID_COL, drop=False)
    out = {}
    for name in ("train", "val", "test"):
        ids = (SPLITS_DIR / f"{name}_ids.txt").read_text(encoding="utf-8").split()
        part = df.loc[ids].copy()
        assert len(part) == len(ids), f"{name}: ids missing from lid.csv"
        part["label_id"] = part[LABEL_COL].map(LABEL2ID)
        out[name] = part
    return out