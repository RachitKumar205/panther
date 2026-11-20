"""Scoring and statistics.

These are the pure functions behind the score, the relative score and the verdict. They fix the
two statistical defects of the old implementation (bug B4):

- the KL standard error is now the standard error of the mean over *chunks* (so correlated text is
  not treated as independent tokens), and
- the standard error of the *difference* between two files is the paired standard error over the
  same chunks (so two files scored on the same text and reference are not treated as independent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import hypot, sqrt

import numpy as np
from numpy.typing import NDArray

Correctness = Mapping[str, Sequence[bool]]
"""Task name -> per-question correctness (0/1 or bool)."""


class Verdict(StrEnum):
    """The outcome of a contest against the preset."""

    WIN = "WIN"
    TIE = "TIE"
    LOSE = "LOSE"


def _as_bool(values: Sequence[bool]) -> NDArray[np.bool_]:
    return np.asarray(values, dtype=np.bool_)


def _mean_bool(values: NDArray[np.bool_]) -> float:
    """Fraction of True values."""
    return float(values.sum() / values.size) if values.size else 0.0


def accuracy(correct: Correctness) -> float:
    """Mean task accuracy, in percent (the old ``acc``)."""
    means = [_mean_bool(_as_bool(correct[task])) for task in correct]
    return 100 * float(np.mean(means)) if means else 0.0


def flips(correct: Correctness, original: Correctness) -> float:
    """Percent of questions whose right/wrong differs from the original, averaged over tasks."""
    rates = [
        _mean_bool(_as_bool(correct[task]) != _as_bool(original[task]))
        for task in original
        if task in correct
    ]
    return 100 * float(np.mean(rates)) if rates else 0.0


def paired_flip_std_error(ours: Correctness, theirs: Correctness, original: Correctness) -> float:
    """Standard error (percent) of the mean flip-difference, paired over questions.

    Per task, the variance of the per-question flip difference is divided by the number of
    questions; tasks are treated as independent and equally weighted.
    """
    variances: list[float] = []
    for task in original:
        if task not in ours or task not in theirs:
            continue
        n = len(original[task])
        if n < 2:
            variances.append(0.0)
            continue
        original_flips = _as_bool(original[task])
        diff = (_as_bool(ours[task]) != original_flips).astype(float) - (
            _as_bool(theirs[task]) != original_flips
        ).astype(float)
        variances.append(float(np.var(diff, ddof=1) / n))
    return 100 * float(sqrt(sum(variances))) / len(original) if variances else 0.0


def kl_mean(chunks: Sequence[float]) -> float:
    """Mean KL over chunks."""
    values = np.asarray(chunks, dtype=np.float64)
    return float(np.mean(values)) if values.size else 0.0


def _std_error(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / sqrt(values.size))


def kl_std_error(chunks: Sequence[float]) -> float:
    """Standard error of the mean KL, from the per-chunk means."""
    return _std_error(np.asarray(chunks, dtype=np.float64))


def kl_diff_std_error(ours: Sequence[float], theirs: Sequence[float]) -> float:
    """Paired standard error of the KL difference over the same chunks."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    if a.shape != b.shape:
        return _std_error(a)
    return _std_error(a - b)


@dataclass(frozen=True, slots=True)
class Score:
    """A file's fitness: flips plus KL, and their standard errors."""

    flips: float
    """Flip rate in percent."""
    kl: float
    """Mean KL divergence."""
    kl_weight: float = 10.0
    """Weight of one unit of KL relative to one point of flips."""
    flips_se: float = 0.0
    """Standard error of the flip rate, percent."""
    kl_se: float = 0.0
    """Standard error of the KL difference from the compared file, in KL units."""

    @property
    def value(self) -> float:
        """The score: ``flips + kl_weight * kl`` (lower is better)."""
        return self.flips + self.kl_weight * self.kl

    @property
    def std_error(self) -> float:
        """Standard error of the score difference from the compared file."""
        return float(hypot(self.flips_se, self.kl_weight * self.kl_se))


def relative_score(ours: float, preset: float) -> float:
    """Our score as a percentage of the preset's (100 = the preset's damage, 0 = none)."""
    return 100 * ours / preset


def relative_interval(
    ours: float,
    preset: float,
    std_error: float,
    z: float = 1.96,
) -> tuple[float, float]:
    """The 95% interval of the relative score: ``100 * (ours +- z * SE) / preset``."""
    half = z * std_error
    return 100 * (ours - half) / preset, 100 * (ours + half) / preset


def verdict(lower: float, upper: float) -> Verdict:
    """WIN if the whole interval is below 100, LOSE if all above, TIE otherwise."""
    if upper < 100:
        return Verdict.WIN
    if lower > 100:
        return Verdict.LOSE
    return Verdict.TIE
