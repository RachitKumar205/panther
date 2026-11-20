"""The per-layer Hessians are symmetric, PSD and the right shapes."""

from __future__ import annotations

import torch

from panther_quant.hessian import compute_hessians
from panther_quant.nn import Qwen2Model


def build() -> Qwen2Model:
    return Qwen2Model(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=50,
    )


def test_hessians_cover_every_layer_with_the_right_shapes() -> None:
    model = build().to(torch.bfloat16).eval()
    ids = torch.randint(0, 50, (4, 8))
    hessians = compute_hessians(model, ids, batch_size=2)
    keys = {(i, name) for i in range(2) for name in ("attn_in", "o_in", "mlp_in", "down_in")}
    assert set(hessians) == keys
    for (_, name), h in hessians.items():
        expected = 32 if name == "down_in" else 16
        assert h.shape == (expected, expected)
        assert torch.allclose(h, h.T)  # symmetric
        assert (torch.diag(h) >= 0).all()  # PSD by construction (x^T x)
