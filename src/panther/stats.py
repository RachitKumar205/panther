"""Paired bootstrap comparison of two configurations on the same test items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACC_MARGIN = 0.005  # 0.5 percentage points
PPL_REL_MARGIN = 0.01  # 1% relative
N_BOOT = 10_000
ALPHA = 0.05


@dataclass(frozen=True)
class Evaluation:
    """Per-item results for one configuration, aligned item-by-item across configurations.

    task_correct: task name -> 0/1 array, one entry per test item (same item order for every config).
    chunk_nll: per-chunk summed negative log-likelihood on the perplexity test text (same chunks).
    chunk_tokens: tokens per chunk.
    """

    name: str
    task_correct: dict[str, np.ndarray]
    chunk_nll: np.ndarray
    chunk_tokens: np.ndarray
    peak_ram_bytes: int
    decode_tps: float

    def mean_accuracy(self) -> float:
        return float(np.mean([v.mean() for v in self.task_correct.values()]))

    def perplexity(self) -> float:
        return float(np.exp(self.chunk_nll.sum() / self.chunk_tokens.sum()))


def _boot_acc_diff(a: Evaluation, b: Evaluation, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap distribution of mean-of-task-accuracies(A) − (B), resampling items within each task."""
    if a.task_correct.keys() != b.task_correct.keys():
        raise ValueError("task sets differ")
    diffs = np.zeros(n_boot)
    for task in a.task_correct:
        d = a.task_correct[task].astype(float) - b.task_correct[task].astype(float)
        if d.shape != b.task_correct[task].shape:
            raise ValueError(f"{task}: item counts differ")
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        diffs += d[idx].mean(axis=1)
    return diffs / len(a.task_correct)


def _boot_log_ppl_ratio(a: Evaluation, b: Evaluation, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap of log(PPL_A / PPL_B), resampling chunks."""
    if not np.array_equal(a.chunk_tokens, b.chunk_tokens):
        raise ValueError("perplexity chunks differ")
    n = len(a.chunk_nll)
    idx = rng.integers(0, n, size=(n_boot, n))
    toks = a.chunk_tokens[idx].sum(axis=1)
    return (a.chunk_nll[idx].sum(axis=1) - b.chunk_nll[idx].sum(axis=1)) / toks


@dataclass(frozen=True)
class Comparison:
    winner: str | None  # name of A or B, or None if indistinguishable on every criterion
    decided_by: str  # "accuracy" | "perplexity" | "ram" | "speed" | "tie"
    acc_diff: float
    acc_ci: tuple[float, float]
    acc_p_one_sided: float  # P(diff <= margin) under the bootstrap, for A beating B
    ppl_ratio: float
    ppl_ratio_ci: tuple[float, float]


def compare(a: Evaluation, b: Evaluation, seed: int = 0, n_boot: int = N_BOOT,
            alpha: float = ALPHA) -> Comparison:
    rng = np.random.default_rng(seed)
    acc = _boot_acc_diff(a, b, rng, n_boot)
    lo, hi = np.quantile(acc, [alpha / 2, 1 - alpha / 2])
    if np.array_equal(a.chunk_tokens, b.chunk_tokens):
        lppl = _boot_log_ppl_ratio(a, b, rng, n_boot)
        plo, phi = np.exp(np.quantile(lppl, [alpha / 2, 1 - alpha / 2]))
        ratio = a.perplexity() / b.perplexity()
    else:  # different tokenizers (models from different families): perplexities are not comparable, so no PPL tiebreak
        plo = phi = ratio = float("nan")
    common: dict[str, Any] = dict(
        acc_diff=a.mean_accuracy() - b.mean_accuracy(),
        acc_ci=(float(lo), float(hi)),
        acc_p_one_sided=float(np.mean(acc <= ACC_MARGIN)),
        ppl_ratio=ratio,
        ppl_ratio_ci=(float(plo), float(phi)),
    )

    if lo > ACC_MARGIN:
        return Comparison(a.name, "accuracy", **common)
    if hi < -ACC_MARGIN:
        return Comparison(b.name, "accuracy", **common)
    if ratio <= 1 - PPL_REL_MARGIN and phi < 1:
        return Comparison(a.name, "perplexity", **common)
    if 1 / ratio <= 1 - PPL_REL_MARGIN and plo > 1:
        return Comparison(b.name, "perplexity", **common)
    if a.peak_ram_bytes != b.peak_ram_bytes:
        w = a.name if a.peak_ram_bytes < b.peak_ram_bytes else b.name
        return Comparison(w, "ram", **common)
    if a.decode_tps != b.decode_tps:
        w = a.name if a.decode_tps > b.decode_tps else b.name
        return Comparison(w, "speed", **common)
    return Comparison(None, "tie", **common)
