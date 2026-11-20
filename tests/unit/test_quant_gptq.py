"""GPTQ compensation reduces the activation-weighted error versus naive rounding."""

from __future__ import annotations

import numpy as np
import torch

from panther_quant.gptq import gptq


def quantize_uniform(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """A per-chunk uniform quantizer (a stand-in for the real block-scale quantizers)."""
    lo, hi = float(w.min()), float(w.max())
    levels = 2**bits - 1
    scale = (hi - lo) / levels
    codes = np.round((w - lo) / scale).astype(np.int32)
    return codes, (lo + codes * scale).astype(np.float32)


def _naive(w: torch.Tensor, bits: int, chunk: int) -> torch.Tensor:
    parts = [
        torch.from_numpy(quantize_uniform(w[:, a : a + chunk].numpy(), bits)[1])
        for a in range(0, w.shape[1], chunk)
    ]
    return torch.cat(parts, dim=1)


def _output_error(w: torch.Tensor, wq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return ((w - wq) @ h @ (w - wq).T).trace()


def test_gptq_reduces_output_error_versus_naive() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 512)
    x = torch.randn(64, 512)
    h = x.T @ x / x.shape[0]
    naive = _naive(w, 4, 256)
    comp = gptq(w, h, lambda a: quantize_uniform(a, 4), 256)
    assert _output_error(w, comp, h) <= _output_error(w, naive, h)


def test_gptq_handles_partial_final_chunk() -> None:
    torch.manual_seed(1)
    w = torch.randn(8, 300)  # 300 is not a multiple of 256
    h = torch.eye(300)
    out = gptq(w, h, lambda a: quantize_uniform(a, 4), 256)
    assert out.shape == (8, 300)
    assert torch.isfinite(out).all()
