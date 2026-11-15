"""Block-AP's row order: llama.cpp's converter reorders attention q/k rows for llama-style models; row_order finds
the file's order from the values, and QLinear trains in that order and hands PyTorch its own."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from types import SimpleNamespace  # noqa: E402

from panther.blockap import row_order  # noqa: E402
from panther.distill import reorder  # noqa: E402

CFG = SimpleNamespace(num_attention_heads=8, num_key_value_heads=4)


@pytest.mark.parametrize("hf,heads", [("self_attn.q_proj", 8), ("self_attn.k_proj", 4)])
def test_reordered_rows_are_found_and_undone(hf, heads):
    w = torch.randn(64, 32)
    file = reorder(w.numpy(), heads) + 0.05 * np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    rows, same, moved = row_order(hf, w, file, CFG)
    assert rows is not None and moved < same
    assert torch.equal(w[rows], torch.from_numpy(reorder(w.numpy(), heads)))  # PyTorch -> the file's order
    assert torch.equal(w[rows][torch.argsort(rows)], w)  # and back


def test_same_rows_are_left():
    w = torch.randn(64, 32)
    assert row_order("self_attn.q_proj", w, w.numpy() + 0.05, CFG)[0] is None  # Qwen: the file keeps PyTorch's order
    assert row_order("mlp.up_proj", w, reorder(w.numpy(), 8), CFG)[0] is None  # only q and k are ever reordered
