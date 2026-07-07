"""P1 main model: XLM-R (frozen) + LoRA on q,v + LID embeddings + SCL head.

Architecture (thesis section 4):
  token embeddings + LID embeddings
    -> XLM-R encoder, frozen, LoRA r=8 alpha=16 on query/value projections
    -> masked mean pooling
    -> classification head (3 classes)  [CE]
    -> projection head, L2-normalized    [SupConLoss]

Trainable: LoRA matrices + LID embedding + classifier + projection head.
Flags allow the ablation variants (no LoRA / no LID / rank / target modules).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from src.common.schema import NUM_CLASSES
from src.models.lid_embedding import LidEmbedding

MODEL_NAME = "xlm-roberta-base"


class ProposedModel(nn.Module):
    def __init__(self, use_lora=True, use_lid=True, lora_r=8, lora_alpha=16,
                 lora_dropout=0.05, target_modules=("query", "value"),
                 proj_dim=128, head_dropout=0.1):
        super().__init__()
        encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = encoder.config.hidden_size

        if use_lora:
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             target_modules=list(target_modules), bias="none")
            self.encoder = get_peft_model(encoder, cfg)  # freezes base weights
        else:
            self.encoder = encoder                        # full fine-tune (ablation)

        self.use_lid = use_lid
        self.lid_emb = LidEmbedding(hidden) if use_lid else None
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden, NUM_CLASSES)
        self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, input_ids, attention_mask, lid_ids=None):
        embeds = self.encoder.get_input_embeddings()(input_ids)
        if self.use_lid and lid_ids is not None:
            embeds = embeds + self.lid_emb(lid_ids)
        out = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        feats = F.normalize(self.proj(pooled), dim=-1)
        return logits, feats