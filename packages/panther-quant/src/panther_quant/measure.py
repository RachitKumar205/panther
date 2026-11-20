"""KL, perplexity and same-top-word from logits (algorithm 05, part 2).

All in float32 (ADR 0019). KL is ``mean over tokens of KL(original || ours)``; every position with
a next token is scored, unlike llama.cpp which reads only the second half of each chunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean next-token cross-entropy: position ``t`` predicts ``labels[t+1]``."""
    logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    labels = labels[:, 1:].reshape(-1)
    return float(F.cross_entropy(logits, labels, reduction="mean"))


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Perplexity: ``exp`` of the mean cross-entropy."""
    return math.exp(cross_entropy(logits, labels))


def kl_divergence(ref_logits: torch.Tensor, logits: torch.Tensor) -> float:
    """Mean per-token KL(original || ours)."""
    ref = F.log_softmax(ref_logits.float(), dim=-1)
    ours = F.log_softmax(logits.float(), dim=-1)
    p = ref.exp()
    return float((p * (ref - ours)).sum(-1).mean())


def same_top_word(ref_logits: torch.Tensor, logits: torch.Tensor) -> float:
    """Percent of positions where the two models' most likely next word agrees."""
    agree = ref_logits.argmax(-1) == logits.argmax(-1)
    return float(agree.float().mean() * 100)
