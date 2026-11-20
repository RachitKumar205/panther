"""The KL reference: the original's top-256 log-probs per token (ADR 0028)."""

from __future__ import annotations

from collections.abc import Callable

import torch

Model = Callable[[torch.Tensor], torch.Tensor]
"""A callable returning ``(batch, seq, vocab)`` logits."""


def topk_logprobs(model: Model, ids: torch.Tensor, k: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """The original's top-k next-token log-probs and ids: ``(logprobs, indices)``."""
    with torch.no_grad():
        top = torch.topk(torch.log_softmax(model(ids).float(), -1), k, -1)
    return top.values, top.indices


def kl_topk(ref_logprobs: torch.Tensor, ref_indices: torch.Tensor, ours_logits: torch.Tensor) -> float:
    """Top-k KL(original || ours): sum over the top-k of ``p_ref (log p_ref - log p_ours)``."""
    log_ps = torch.log_softmax(ours_logits.float(), dim=-1)
    p_ref = ref_logprobs.float().exp()
    gathered = log_ps.gather(-1, ref_indices.long())
    return float((p_ref * (ref_logprobs.float() - gathered)).sum(-1).mean())
