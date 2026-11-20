"""Offline quantization error (bug B3).

The corrected formula: per-column squared error weighted by that column's input importance. The old
code summed over input columns and weighted by a truncated importance vector, which used the wrong
axis and the wrong length for non-square matrices.
"""

from __future__ import annotations

import numpy as np


def weighted_sqerr(w: np.ndarray, wq: np.ndarray, importance: np.ndarray) -> float:
    """``sum over columns of importance[c] * sum over rows (w - wq)^2``.

    Args:
        w: The original matrix, ``(rows, cols)``.
        wq: The quantized (dequantized) matrix, same shape as ``w``.
        importance: One value per input column (length ``cols``).

    Raises:
        ValueError: If the shapes do not line up.
    """
    a = np.asarray(w, dtype=np.float64)
    q = np.asarray(wq, dtype=np.float64)
    imp = np.asarray(importance, dtype=np.float64)
    if a.ndim != 2 or a.shape != q.shape or imp.shape != (a.shape[1],):
        raise ValueError("w and wq must be (rows, cols) and importance must have cols elements")
    return float(((a - q) ** 2).sum(axis=0) @ imp)
