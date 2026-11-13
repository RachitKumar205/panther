"""Paired bootstrap: the difference in mean accuracy (mean over tasks of each task's accuracy) between two files
answered on the same questions, with its 95% interval. WIN / LOSE when the whole interval is above / below 0."""

from __future__ import annotations

import numpy as np


def acc(correct: dict[str, np.ndarray]) -> float:
    return 100 * float(np.mean([np.mean(v) for v in correct.values()]))


def flips(correct: dict[str, np.ndarray], original: dict[str, np.ndarray]) -> float:
    """% of questions whose right/wrong differs from the original's, averaged over tasks."""
    return 100 * float(np.mean([np.mean(correct[t] != original[t]) for t in original]))


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n: int = 10_000) -> tuple[float, float, float]:
    """(a - b, low, high) in accuracy points, resampling questions within each task."""
    rng = np.random.default_rng(0)
    boot = np.zeros(n)
    for t in a:
        d = a[t].astype(float) - b[t].astype(float)
        boot += d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    boot *= 100 / len(a)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return acc(a) - acc(b), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    return "WIN" if lo > 0 else "LOSE" if hi < 0 else "TIE"
