"""GPTQ: the maths with a simple stand-in quantizer, then with llama.cpp's own quantizer for every type."""

from __future__ import annotations

import gguf
import numpy as np
import pytest
import torch

from panther.model.gptq import GPTQ
from panther.model.model import TYPES
from panther.model.quantizer import Quantizer
from tests.model.tiny import needs_ggml, random_hessian

ROWS, COLS = 48, 768


class Grid:
    """A stand-in quantizer: each row onto 15 levels of its own scale."""

    def quantize(self, w: np.ndarray, importance: np.ndarray, type_: str) -> tuple[np.ndarray, np.ndarray]:
        scale = np.abs(w).max(axis=1, keepdims=True) / 7 + 1e-12
        q = np.clip(np.round(w / scale), -7, 7)
        return q.astype(np.int8).view(np.uint8), (q * scale).astype(np.float32)


class Recording:
    """Wraps a quantizer and records every call: the importance it was given, the codes and values it returned."""

    def __init__(self, quantizer) -> None:
        self.quantizer, self.importance, self.codes, self.values = quantizer, [], [], []

    def quantize(self, w: np.ndarray, importance: np.ndarray, type_: str) -> tuple[np.ndarray, np.ndarray]:
        codes, values = self.quantizer.quantize(w, importance, type_)
        self.importance.append(importance.copy())
        self.codes.append(codes)
        self.values.append(values)
        return codes, values


def output_error(w: np.ndarray, q: np.ndarray, h: np.ndarray) -> float:
    """How much the matrix's output changes over the inputs H describes: trace((W-Q) H (W-Q)ᵀ)."""
    d = (w - q).astype(np.float64)
    return float(np.einsum("ij,jk,ik->", d, h, d))


def plain(w: np.ndarray, h: np.ndarray, quantizer, type_: str) -> np.ndarray:
    """Every block rounded on its own, no compensation."""
    return np.concatenate([quantizer.quantize(w[:, a:a + 256], np.diag(h)[a:a + 256], type_)[1]
                           for a in range(0, COLS, 256)], axis=1)


def gptq(w: np.ndarray, h: np.ndarray, quantizer, type_: str) -> tuple[np.ndarray, Recording]:
    """GPTQ's codes, and the record of what it asked the quantizer."""
    recording = Recording(quantizer)
    return GPTQ(torch.from_numpy(h)).round(torch.from_numpy(w), type_, recording), recording


@pytest.fixture
def case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32), random_hessian(COLS, 1)


def test_with_independent_inputs_gptq_is_plain_rounding(case) -> None:
    w, _ = case
    h = np.diag(np.linspace(0.5, 2, COLS)).astype(np.float32)  # no correlation: nothing to compensate
    _, used = gptq(w, h, Grid(), "x")
    assert np.allclose(np.concatenate(used.values, axis=1), plain(w, h, Grid(), "x"), atol=1e-5)


def test_compensation_lowers_the_output_error(case) -> None:
    w, h = case
    _, used = gptq(w, h, Grid(), "x")
    assert output_error(w, np.concatenate(used.values, axis=1), h) < 0.8 * output_error(w, plain(w, h, Grid(), "x"), h)


def test_each_block_gets_its_own_columns_importance_and_its_codes_are_returned(case) -> None:
    w, h = case
    codes, used = gptq(w, h, Grid(), "x")
    assert len(used.importance) == COLS // 256
    for i, importance in enumerate(used.importance):
        assert np.allclose(importance, np.diag(h)[i * 256:(i + 1) * 256])
    assert np.array_equal(codes, np.concatenate(used.codes, axis=1))


def test_a_column_never_used_is_zeroed(case) -> None:
    w, h = case
    h = h.copy()
    h[5, :], h[:, 5] = 0, 0
    codes, _ = gptq(w, h, Grid(), "x")
    assert (codes.view(np.int8)[:, 5] == 0).all()


@needs_ggml
@pytest.mark.parametrize("type_", TYPES)
def test_llama_cpp_codes_decode_to_the_values_gptq_used(case, type_: str) -> None:
    # llama.cpp's codes, decoded by gguf-py (an independent decoder), are exactly the values GPTQ compensated for
    w, h = case
    codes, used = gptq(w, h, Quantizer(), type_)
    decoded = gguf.quants.dequantize(codes, gguf.GGMLQuantizationType[type_])
    assert np.array_equal(decoded, np.concatenate(used.values, axis=1))


@needs_ggml
@pytest.mark.parametrize("type_", ["IQ2_XS", "Q2_K", "Q3_K", "IQ4_XS", "Q4_K"])
def test_llama_cpp_gptq_beats_plain_rounding(case, type_: str) -> None:
    w, h = case
    _, used = gptq(w, h, Quantizer(), type_)
    assert output_error(w, np.concatenate(used.values, axis=1), h) < output_error(w, plain(w, h, Quantizer(), type_), h)
