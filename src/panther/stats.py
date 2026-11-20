"""Accuracy, flips, and the relative score with its 95% interval: one file's score (flips % + kl_weight x KL) as a
share of another's (x 100), both answered on the same questions. WIN / LOSE when the whole interval is below / above
100 (lower is better)."""

from __future__ import annotations

import numpy as np


def acc(correct: dict[str, np.ndarray]) -> float:
    return 100 * float(np.mean([np.mean(v) for v in correct.values()]))


def flips(correct: dict[str, np.ndarray], original: dict[str, np.ndarray]) -> float:
    """% of questions whose right/wrong differs from the original's, averaged over tasks."""
    return 100 * float(np.mean([np.mean(correct[t] != original[t]) for t in original]))


def score_se(a: dict[str, np.ndarray], b: dict[str, np.ndarray], original: dict[str, np.ndarray], kl_se_a: float,
             kl_se_b: float, kl_weight: float) -> float:
    """Standard error of the difference of two scores (flips % + kl_weight x KL). Flips: paired over the same
    questions, per task the SE of the mean difference of the flip indicators, combined as the mean over tasks;
    KL: the two runs' own SEs (llama-perplexity's), independent."""
    f = 100 * np.sqrt(sum(np.var((a[t] != o).astype(float) - (b[t] != o).astype(float), ddof=1) / len(o)
                          for t, o in original.items())) / len(original)
    return float(np.hypot(f, kl_weight * np.hypot(kl_se_a, kl_se_b)))


def relative(a: dict[str, np.ndarray], b: dict[str, np.ndarray], original: dict[str, np.ndarray], kl_a: float,
             kl_se_a: float, kl_b: float, kl_se_b: float, kl_weight: float) -> tuple[float, float, float]:
    """a's score as % of b's, with its 95% interval: 100 x (score_a +- 1.96 SE of the difference) / score_b."""
    sa, sb = flips(a, original) + kl_weight * kl_a, flips(b, original) + kl_weight * kl_b
    h = 1.96 * score_se(a, b, original, kl_se_a, kl_se_b, kl_weight)
    return 100 * sa / sb, 100 * (sa - h) / sb, 100 * (sa + h) / sb


def verdict(lo: float, hi: float) -> str:
    """From a relative score's interval: below 100 is better than the other file."""
    return "WIN" if hi < 100 else "LOSE" if lo > 100 else "TIE"
