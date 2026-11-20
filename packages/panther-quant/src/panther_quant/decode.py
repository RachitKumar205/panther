"""Greedy decoding: take the argmax next token repeatedly (for GSM8K and chat answers)."""

from __future__ import annotations

from collections.abc import Callable

import torch

Model = Callable[[torch.Tensor], torch.Tensor]
"""A callable returning ``(batch, seq, vocab)`` logits."""


def greedy_decode(model: Model, input_ids: torch.Tensor, max_tokens: int, eos_id: int | None = None) -> torch.Tensor:
    """Append the argmax next token up to ``max_tokens`` times, stopping early at ``eos_id``."""
    ids = input_ids
    for _ in range(max_tokens):
        next_token = model(ids)[:, -1].argmax(-1, keepdim=True)  # (batch, 1)
        ids = torch.cat([ids, next_token], dim=-1)
        if eos_id is not None and bool((next_token == eos_id).all()):
            break
    return ids
