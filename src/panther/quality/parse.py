"""Parsers for llama-perplexity output at the pinned llama.cpp commit (docs/PIPELINE.md A3).

llama-perplexity exits 0 even when it fails, so a parse that finds no summary is a failed run.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


class ParseError(ValueError):
    """The output lacks what a successful run prints."""


def scored_tokens_per_chunk(n_ctx: int) -> int:
    """llama-perplexity scores tokens first..n_ctx-2 of each chunk, first = n_ctx/2 (perplexity.cpp)."""
    return n_ctx - n_ctx // 2 - 1


_KL = {
    "ppl_q": re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
    "ppl_ratio": re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)\s*±\s*([0-9.na-]+)"),
    "kl_mean": re.compile(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*±\s*([0-9.]+)"),
    "kl_p99": re.compile(r"99\.0%\s+KLD:\s*(-?[0-9.]+)"),
    "kl_max": re.compile(r"Maximum KLD:\s*(-?[0-9.]+)"),
    "top1": re.compile(r"Same top p:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*%"),
}
_KL_CHUNKS = re.compile(r"kl_divergence: computing over (\d+) chunks, n_ctx=(\d+)")


@dataclass(frozen=True)
class KLResult:
    kl_mean: float
    kl_mean_se: float
    kl_p99: float
    kl_max: float
    top1_pct: float
    top1_pct_se: float
    ppl_q: float
    ppl_ratio: float
    chunks: int
    n_ctx: int


def parse_kl(text: str) -> KLResult:
    vals: dict[str, tuple[float, float | None]] = {}
    for key, rx in _KL.items():
        m = rx.search(text)
        if m is None:
            raise ParseError(f"KL output has no '{key}' line")
        second = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        try:
            se = float(second) if second is not None else None
        except ValueError:  # "-nan" when the ratio has no spread
            se = None
        vals[key] = (float(m.group(1)), se)
    c = _KL_CHUNKS.search(text)
    if c is None:
        raise ParseError("KL output has no chunk count line")
    return KLResult(
        kl_mean=vals["kl_mean"][0], kl_mean_se=vals["kl_mean"][1] or 0.0,
        kl_p99=vals["kl_p99"][0], kl_max=vals["kl_max"][0],
        top1_pct=vals["top1"][0], top1_pct_se=vals["top1"][1] or 0.0,
        ppl_q=vals["ppl_q"][0], ppl_ratio=vals["ppl_ratio"][0],
        chunks=int(c.group(1)), n_ctx=int(c.group(2)),
    )


_PPL_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.M)
_PPL_FINAL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
_PPL_CHUNKS = re.compile(r"perplexity: calculating perplexity over (\d+) chunks, n_ctx=(\d+)")

RECOVERY_TOLERANCE = 1e-4  # recovered per-chunk NLL must reproduce the final perplexity within 0.01%


@dataclass(frozen=True)
class PPLResult:
    ppl: float
    ppl_se: float
    chunk_mean_nll: np.ndarray  # per-chunk mean NLL per scored token, recovered from running means
    tokens_per_chunk: int
    n_ctx: int
    recovery_rel_error: float

    @property
    def chunk_nll(self) -> np.ndarray:
        """Summed NLL per chunk, as panther.stats.decision.Evaluation expects."""
        return self.chunk_mean_nll * self.tokens_per_chunk

    @property
    def chunk_tokens(self) -> np.ndarray:
        return np.full(len(self.chunk_mean_nll), self.tokens_per_chunk)


def parse_ppl_type1(text: str) -> PPLResult:
    """Parse `llama-perplexity --ppl-output-type 1`: one row per chunk with the running mean NLL to
    6 decimals. Chunk i's own mean is i*m_i - (i-1)*m_(i-1), since every chunk scores the same count."""
    head = _PPL_CHUNKS.search(text)
    final = _PPL_FINAL.search(text)
    if head is None or final is None:
        raise ParseError("perplexity output lacks the chunk-count or final-estimate line")
    n_chunks, n_ctx = int(head.group(1)), int(head.group(2))
    rows = [(int(m.group(1)), float(m.group(3))) for m in _PPL_ROW.finditer(text)]
    if len(rows) != n_chunks:
        raise ParseError(f"expected {n_chunks} per-chunk rows, found {len(rows)} (missing --ppl-output-type 1?)")
    for i, (pos, _) in enumerate(rows):
        if pos != i * n_ctx:
            raise ParseError(f"row {i} at position {pos}, expected {i * n_ctx}")
    running = np.array([r[1] for r in rows])
    idx = np.arange(1, n_chunks + 1)
    totals = running * idx
    chunk_mean = np.diff(np.concatenate([[0.0], totals]))
    ppl = float(final.group(1))
    recovered = math.exp(chunk_mean.mean())
    rel = abs(recovered - ppl) / ppl
    if rel > RECOVERY_TOLERANCE:
        raise ParseError(f"recovered per-chunk NLL gives PPL {recovered:.6f}, llama.cpp says {ppl} ({rel:.2e})")
    return PPLResult(ppl=ppl, ppl_se=float(final.group(2)), chunk_mean_nll=chunk_mean,
                     tokens_per_chunk=scored_tokens_per_chunk(n_ctx), n_ctx=n_ctx, recovery_rel_error=rel)


_TYPE_LINE = re.compile(r"- type\s+(\S+):\s+(\d+) tensors")


def parse_type_counts(text: str) -> dict[str, int]:
    """Per-type tensor counts from the loader log (printed at -lv 4 and above)."""
    return {m.group(1).lower(): int(m.group(2)) for m in _TYPE_LINE.finditer(text)}


# --- task modes (A4, D38) ---------------------------------------------------------------------------------

_MC_ROW = re.compile(r"^(\d+)\t(\d+\.\d{8})$", re.M)
_MC_FINAL = re.compile(r"Final result: ([0-9.]+) \+/- ([0-9.]+)")
_WG_ROW = re.compile(r"^(\d+)\t([0-9.]+)\t\s*(\S+)\s+(\S+)\s+([12])\s+([12])$", re.M)
_WG_FINAL = re.compile(r"Final Winogrande score\((\d+) tasks\): ([0-9.]+) \+/- ([0-9.]+)")


@dataclass(frozen=True)
class TaskResult:
    correct: np.ndarray  # 0/1 per question, in file order
    accuracy_pct: float  # llama.cpp's final figure
    accuracy_se_pct: float


def parse_multiple_choice(text: str, n_expected: int) -> TaskResult:
    """`--multiple-choice` prints the running accuracy (8 decimals) after each question; the running count of
    correct answers is round(acc * i / 100), and its steps are the per-question results."""
    rows = [(int(m.group(1)), float(m.group(2))) for m in _MC_ROW.finditer(text)]
    final = _MC_FINAL.search(text)
    if final is None or len(rows) != n_expected:
        raise ParseError(f"multiple-choice output: {len(rows)} rows of {n_expected}, final line "
                         f"{'present' if final else 'missing'}")
    counts = []
    for k, (i, acc) in enumerate(rows, 1):
        if i != k:
            raise ParseError(f"row {k} numbered {i}")
        c = acc * i / 100
        if abs(c - round(c)) > 1e-3:
            raise ParseError(f"row {i}: running count {c} is not an integer")
        counts.append(round(c))
    steps = np.diff(np.concatenate([[0], counts]))
    if not set(np.unique(steps)) <= {0, 1}:
        raise ParseError("running count moved by more than one")
    acc = float(final.group(1))
    if abs(100 * steps.mean() - acc) > 1e-3:
        raise ParseError(f"recovered accuracy {100 * steps.mean():.4f} != reported {acc}")
    return TaskResult(steps.astype(np.int8), acc, float(final.group(2)))


def parse_winogrande(text: str, n_expected: int) -> TaskResult:
    """`--winogrande` prints, per question, the choice made and the answer."""
    rows = [(int(m.group(1)), int(m.group(5)), int(m.group(6))) for m in _WG_ROW.finditer(text)]
    final = _WG_FINAL.search(text)
    if final is None or len(rows) != n_expected or int(final.group(1)) != n_expected:
        raise ParseError(f"winogrande output: {len(rows)} rows of {n_expected}, final line "
                         f"{'present' if final else 'missing'}")
    if [r[0] for r in rows] != list(range(1, n_expected + 1)):
        raise ParseError("winogrande rows out of order")
    correct = np.array([int(r[1] == r[2]) for r in rows], dtype=np.int8)
    acc = float(final.group(2))
    if abs(100 * correct.mean() - acc) > 1e-3:
        raise ParseError(f"recovered accuracy {100 * correct.mean():.4f} != reported {acc}")
    return TaskResult(correct, acc, float(final.group(3)))
