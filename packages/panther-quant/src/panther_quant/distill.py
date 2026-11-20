"""Distillation: block-scale multipliers and the top-k KL loss (algorithm 09)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockScale(nn.Module):
    """Multiply each 256-weight block of a weight by ``exp(theta)``, ``theta`` in fp32."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(out_features, in_features // 256))  # one multiplier per block

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scaled = w.float().view(w.shape[0], -1, 256) * torch.exp(self.theta)[:, :, None]
        return scaled.view(w.shape).to(w.dtype)


def kl_topk_loss(
    logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(teacher top-k || student) over positions.

    Args:
        logits: Student logits ``(..., vocab)``.
        teacher_values: Teacher probabilities ``(..., k)``, renormalized over the top-k.
        teacher_indices: Teacher token ids ``(..., k)``.
    """
    log_ps = torch.log_softmax(logits.float(), dim=-1)  # (..., vocab)
    p_t = teacher_values.float() / teacher_values.float().sum(-1, keepdim=True)
    log_p_t = torch.log(p_t + 1e-12)  # avoid log(0) -> -inf
    gathered = log_ps.gather(-1, teacher_indices.long())  # (..., k)
    return (p_t * (log_p_t - gathered)).sum(-1).mean()
