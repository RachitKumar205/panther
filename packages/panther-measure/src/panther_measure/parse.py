"""Parsers for llama-perplexity output.

The KL parser keeps the per-chunk means, so the paired-difference standard error (bug B4) can be
computed from the chunks of two files scored on the same text.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, ConfigDict


class KlMetrics(BaseModel):
    """One KL run's summary and its per-chunk means."""

    model_config = ConfigDict(frozen=True)

    mean: float  # mean KL divergence
    se: float  # llama.cpp's reported SE over chunks
    top1_pct: float  # same top word percent
    chunks: tuple[float, ...]  # per-chunk KL means, for the paired-difference SE


class ParseError(ValueError):
    """llama.cpp output did not contain what was expected."""


def parse_kl(text: str) -> KlMetrics:
    m = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*±\s*([0-9.]+)", text)
    top = re.search(r"Same top p:\s*([0-9.]+)", text)
    if not (m and top):
        raise ParseError("KL output incomplete")
    chunk_re = r"^\s*\d+\s+[0-9.]+\s*±\s*[0-9.]+\s+[0-9.]+\s*±\s*[0-9.]+\s+([0-9.]+)\s*±"
    return KlMetrics(
        mean=float(m[1]),
        se=float(m[2]),
        top1_pct=float(top[1]),
        chunks=tuple(float(c) for c in re.findall(chunk_re, text, re.M)),
    )


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
