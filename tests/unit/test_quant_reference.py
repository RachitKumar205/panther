"""The top-k KL reference is zero for identical distributions and non-negative otherwise."""

from __future__ import annotations

import pytest
import torch

from panther_quant.nn import Qwen2Model
from panther_quant.reference import kl_topk, topk_logprobs


def build() -> Qwen2Model:
    return Qwen2Model(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=50,
    )


def test_kl_topk_is_zero_for_identical_distributions() -> None:
    model = build().eval()
    ids = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        logits = model(ids).float()
    logprobs, indices = topk_logprobs(model, ids, k=16)
    assert kl_topk(logprobs, indices, logits) == pytest.approx(0.0, abs=1e-4)


def test_kl_topk_is_nonnegative_and_measures_divergence() -> None:
    a = torch.randn(2, 5, 10)
    b = torch.randn(2, 5, 10)
    logprobs, indices = topk_logprobs(lambda x: a, torch.zeros(2, 5), k=8)
    assert kl_topk(logprobs, indices, b) >= 0
