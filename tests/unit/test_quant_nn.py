"""The Qwen2 forward pass is structurally sound and deterministic."""

from __future__ import annotations

import torch

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


def test_logits_shape_dtype_and_finiteness() -> None:
    model = build().to(torch.bfloat16).eval()
    ids = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 8, 50)
    assert logits.dtype == torch.float32  # upcast before softmax
    assert torch.isfinite(logits).all()


def test_forward_is_deterministic() -> None:
    model = build().to(torch.bfloat16).eval()
    ids = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        assert torch.equal(model(ids), model(ids))


def test_causal_mask_blocks_future_tokens() -> None:
    model = build().to(torch.bfloat16).eval()
    seq = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        logits = model(seq)
    # Changing only the final token leaves the earlier positions' logits untouched.
    seq2 = seq.clone()
    seq2[0, -1] = 7
    with torch.no_grad():
        logits2 = model(seq2)
    assert torch.allclose(logits[:, :-1], logits2[:, :-1])
