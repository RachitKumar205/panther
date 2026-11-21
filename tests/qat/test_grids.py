"""The 13 types in PyTorch against llama.cpp's own files: bytes back byte for byte, decoding bit for bit, codes picked
the same whatever the precision around them, gradients straight through."""

from __future__ import annotations

import gguf
import numpy as np
import pytest
import torch

from panther.distill.distill import SCALE_AT
from panther.model import TYPES
from panther.model.quantizer import Quantizer
from panther.qat.grids import SPECS, Grid
from tests.model.tiny import needs_ggml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def random_bytes(t: str, rows: int = 8, blocks: int = 4, seed: int = 0) -> np.ndarray:
    """Valid random blocks: any codes, scales and signs, and a finite 16-bit d (and dmin)."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, (rows, blocks, SPECS[t].block_bytes), dtype=np.uint8)
    d = (rng.standard_normal((rows, blocks, 2)) * 1e-2).astype(np.float16)
    if t == "IQ1_M":  # d is spread over the top 4 bits of its 4 scale words
        w = b[..., 48:56].copy().view(np.uint16)
        nib = (d[..., 0].view(np.uint16)[..., None] >> (np.arange(4, dtype=np.uint16) * 4)) & 15
        b[..., 48:56] = ((w & 4095) | (nib << 12)).astype(np.uint16).view(np.uint8)
    else:
        for k, at in enumerate(SCALE_AT[t]):
            b[..., at:at + 2] = d[..., k:k + 1].view(np.uint8)
    return b.reshape(rows, -1)


def test_every_type_of_ours_and_no_other() -> None:
    assert set(SPECS) == set(TYPES)


@pytest.mark.parametrize("t", TYPES)
def test_bytes_come_back_exactly_and_decode_bit_for_bit(t: str) -> None:
    raw, grid = random_bytes(t), Grid(t)
    state = grid.unpack(raw)
    ours = grid.decode(state).numpy()
    reference = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType[t]).reshape(ours.shape)
    assert np.array_equal(ours.view(np.uint32), reference.view(np.uint32))  # bit for bit, -0.0 included
    assert np.array_equal(grid.pack(state), raw)  # every byte back


@needs_ggml
@pytest.mark.parametrize("t", TYPES)
def test_a_llama_cpp_files_values_encode_to_the_same_values(t: str) -> None:
    w = np.random.default_rng(1).standard_normal((32, 1024)).astype(np.float32) * 0.02
    codes, values = Quantizer().quantize(w, np.ones(1024, np.float32), t)
    grid = Grid(t)
    state = grid.unpack(codes, DEVICE)
    again = grid.decode({**state, **grid.encode(grid.decode(state), state)})
    assert torch.equal(again.cpu(), torch.from_numpy(values))  # the file's own values are their own nearest codes


@pytest.mark.parametrize("t", TYPES)
def test_codes_are_picked_in_32_bits_whatever_the_precision_around(t: str) -> None:
    grid = Grid(t)
    state = grid.unpack(random_bytes(t, 32, 8), DEVICE)
    w = grid.decode(state)
    w = w + torch.randn(w.shape, generator=torch.Generator().manual_seed(0)).to(w.device) * w.std() * 0.05
    plain = grid.encode(w, state)
    with torch.autocast(DEVICE, torch.bfloat16):
        cast = grid.encode(w, state)
    assert all(torch.equal(plain[k], cast[k]) for k in plain)


@pytest.mark.parametrize("t", TYPES)
def test_the_forward_pass_is_exact_the_gradient_straight_through_and_the_result_a_valid_file(t: str) -> None:
    grid = Grid(t)
    state = grid.unpack(random_bytes(t))
    w = grid.decode(state)
    w = (w + torch.randn(w.shape, generator=torch.Generator().manual_seed(1)) * 1e-3).requires_grad_(True)
    out = grid.fake(w, state)
    exact = grid.decode({**state, **grid.encode(w, state)})
    assert torch.equal(out.detach(), exact)
    (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(2))).sum().backward()
    assert bool((w.grad != 0).all())  # every weight gets the gradient, as if there were no rounding
    packed = grid.pack({**state, **grid.encode(w, state)})
    assert np.array_equal(grid.decode(grid.unpack(packed)).numpy(), exact.numpy())
