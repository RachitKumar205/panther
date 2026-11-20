"""Exact quantization byte accounting."""

from __future__ import annotations

import pytest

from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType


@pytest.mark.parametrize(
    ("name", "bits"),
    [
        ("IQ1_S", 1.5625),
        ("IQ2_XS", 2.3125),
        ("Q2_K", 2.625),
        ("Q3_K", 3.4375),
        ("Q4_K", 4.5),
        ("Q8_0", 8.5),
        ("BF16", 16.0),
        ("F32", 32.0),
    ],
)
def test_bits_per_weight(name: str, bits: float) -> None:
    assert QuantType[name].bits_per_weight == pytest.approx(bits)


def test_search_formats_are_ordered_low_to_high() -> None:
    bits = [quant.bits_per_weight for quant in SEARCH_FORMATS]
    assert bits == sorted(bits)
    assert len(SEARCH_FORMATS) == 14


def test_q2k_uses_the_real_bits_per_weight() -> None:
    # The old PRESETS table listed Q2_K as 3.2 bits/weight; the format is 2.625 (bug B15).
    assert QuantType.Q2_K.bits_per_weight == pytest.approx(2.625)


def test_nbytes_is_rows_times_blocks_times_block_size() -> None:
    assert QuantType.Q4_K.nbytes(rows=10, cols=512) == 10 * 2 * 144
    assert QuantType.Q8_0.nbytes(rows=3, cols=64) == 3 * 2 * 34


def test_nbytes_rejects_columns_that_do_not_fit_the_block() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        QuantType.Q4_K.nbytes(rows=1, cols=100)


def test_nbytes_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError, match="negative"):
        QuantType.Q4_K.nbytes(rows=-1, cols=256)


def test_imatrix_formats_are_a_subset_of_the_search_formats() -> None:
    assert set(SEARCH_FORMATS) >= IMATRIX_FORMATS
