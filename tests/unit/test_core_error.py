"""Offline quantization error weights each input column (bug B3)."""

from __future__ import annotations

import numpy as np
import pytest

from panther_core.error import weighted_sqerr


def test_weighted_sqerr_weights_each_input_column() -> None:
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    wq = w.copy()
    wq[:, 0] += 1.0  # error of 1 in both rows of column 0 only
    importance = np.array([10.0, 0.0, 0.0])
    assert weighted_sqerr(w, wq, importance) == pytest.approx(2 * 1.0 * 10.0)


def test_weighted_sqerr_is_sum_over_columns_of_weighted_row_error() -> None:
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    wq = np.array([[1.0, 0.0], [0.0, 4.0]])  # column errors: (0,3) and (2,0) -> squared sums 9 and 4
    importance = np.array([2.0, 5.0])
    assert weighted_sqerr(w, wq, importance) == pytest.approx(2.0 * 9.0 + 5.0 * 4.0)


def test_weighted_sqerr_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        weighted_sqerr(np.ones((2, 3)), np.ones((3, 2)), np.ones(3))
