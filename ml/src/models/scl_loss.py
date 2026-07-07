"""Supervised Contrastive Loss (Khosla et al. 2020), reimplemented.

Operates on L2-normalized feature vectors. Positives = same-label samples in
the batch (self excluded). loss = -mean over anchors of mean log-prob of
positives under temperature-scaled cosine similarities.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """features: (B, D) L2-normalized. labels: (B,) int."""
        device = features.device
        sim = features @ features.T / self.temperature          # (B, B)
        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)      # same label
        self_mask = torch.eye(labels.size(0), device=device)
        pos_mask = pos_mask - self_mask                          # exclude self

        exp_sim = torch.exp(sim) * (1 - self_mask)              # exclude self
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)                          # anchors w/ >=1 positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        loss = -mean_log_prob_pos[pos_count > 0]
        return loss.mean() if loss.numel() > 0 else features.new_zeros(())