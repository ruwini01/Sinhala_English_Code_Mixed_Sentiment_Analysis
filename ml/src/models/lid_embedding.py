"""LID embedding layer: per-token language tag -> hidden-size vector, added
to the XLM-R word embeddings. Zero-initialized so training starts exactly at
baseline behavior and learns language-tag offsets only if useful.

Tag ids come from schema.LID_TAGS: sin=0, eng=1, other=2, special=3.
"""

import torch.nn as nn

from src.common.schema import LID_TAGS


class LidEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(len(LID_TAGS), hidden_size)
        nn.init.zeros_(self.emb.weight)

    def forward(self, lid_ids):
        return self.emb(lid_ids)