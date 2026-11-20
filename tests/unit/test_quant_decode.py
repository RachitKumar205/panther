"""Greedy decoding appends the argmax token and stops at EOS."""

from __future__ import annotations

import torch

from panther_quant.decode import greedy_decode
from panther_quant.nn import Qwen2Model


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


def test_greedy_decode_appends_the_argmax_next_token() -> None:
    model = build().eval()
    ids = torch.randint(0, 50, (1, 4))
    out = greedy_decode(model, ids, max_tokens=3)
    assert out.shape == (1, 7)
    with torch.no_grad():
        expected = model(ids)[:, -1].argmax(-1, keepdim=True)
    assert torch.equal(out[:, 4], expected[:, 0])


def test_greedy_decode_stops_at_eos() -> None:
    def model(ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(ids.shape[0], ids.shape[1], 5)
        logits[:, :, 3] = 100.0  # always predict token 3
        return logits

    out = greedy_decode(model, torch.tensor([[0, 1]]), max_tokens=5, eos_id=3)
    assert out.shape == (1, 3)  # 2 input tokens + one EOS
