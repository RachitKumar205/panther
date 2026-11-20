"""Parsers for llama-perplexity output.

The KL parser recovers the true per-chunk means (llama-perplexity prints a running average, not a
chunk-wise value) so the paired-difference standard error (bug B4) is computed over real chunks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


class KlMetrics(BaseModel):
    """One KL run's summary and its recovered per-chunk means."""

    model_config = ConfigDict(frozen=True)

    mean: float  # mean KL divergence (the summary, not recomputed from chunks)
    se: float  # llama.cpp's reported SE over tokens (documented as too small; kept for reference)
    top1_pct: float  # same top word percent
    chunks: tuple[float, ...]  # recovered per-chunk KL means, for the paired-difference SE


class ParseError(ValueError):
    """llama.cpp output did not contain what was expected."""


def parse_kl(text: str) -> KlMetrics:
    m = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*±\s*([0-9.]+)", text)
    top = re.search(r"Same top p:\s*([0-9.]+)", text)
    if not (m and top):
        raise ParseError("KL output incomplete")
    chunk_re = r"^\s*\d+\s+[0-9.]+\s*±\s*[0-9.]+\s+[0-9.]+\s*±\s*[0-9.]+\s+([0-9.]+)\s*±"
    cumulative = [float(c) for c in re.findall(chunk_re, text, re.M)]
    return KlMetrics(
        mean=float(m[1]),
        se=float(m[2]),
        top1_pct=float(top[1]),
        chunks=_unchunk(cumulative),
    )


def _unchunk(cumulative: Sequence[float]) -> tuple[float, ...]:
    """Recover per-chunk KL from the running average: ``i*cum[i] - (i-1)*cum[i-1]``."""
    per_chunk: list[float] = []
    prev = 0.0
    for i, total in enumerate(cumulative, 1):
        per_chunk.append(i * total - (i - 1) * prev)
        prev = total
    return tuple(per_chunk)


def parse_ppl(text: str) -> float:
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if not m:
        raise ParseError("no final perplexity")
    return float(m[1])


def parse_choice(text: str, n: int) -> np.ndarray:
    """Per-question 0/1, reconstructed from llama-perplexity's running accuracy."""
    rows = [(int(a), float(b)) for a, b in re.findall(r"^(\d+)\t(\d+\.\d{8})$", text, re.M)]
    if [i for i, _ in rows] != list(range(1, n + 1)):
        raise ParseError(f"{len(rows)} of {n} questions")
    counts = np.array([acc * i / 100 for i, acc in rows]).round().astype(int)
    return np.diff(np.concatenate([[0], counts])).astype(np.int8)


def parse_winogrande(text: str, n: int) -> np.ndarray:
    rows = re.findall(r"^\d+\t[0-9.]+\t\s*\S+\s+\S+\s+([12])\s+([12])$", text, re.M)
    if len(rows) != n:
        raise ParseError(f"{len(rows)} of {n} questions")
    return np.array([int(a == b) for a, b in rows], dtype=np.int8)
