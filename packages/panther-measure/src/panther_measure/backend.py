"""The llama.cpp seam.

Every measurement goes through a :class:`LlamaBackend`. The real implementation drives
``llama-perplexity``/``llama-server`` subprocesses; the fake returns deterministic results, so the
pipeline is testable without a GPU or the llama.cpp binaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from panther_measure.parse import KlMetrics


class LlamaBackend(Protocol):
    """Runs llama.cpp measurements on a GGUF file."""

    def kl(self, model: Path, reference: Path, ctx: int) -> KlMetrics:
        """KL divergence against a recorded reference, over ``ctx``-token chunks."""
        raise NotImplementedError

    def ppl(self, model: Path, text: Path, ctx: int) -> float:
        """Perplexity on ``text``."""
        raise NotImplementedError

    def choice(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        """Multiple-choice correctness (0/1 per question)."""
        raise NotImplementedError

    def winogrande(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        """Winograd-style correctness (0/1 per question)."""
        raise NotImplementedError

    def complete(self, model: Path, prompts: Sequence[str], chat: bool) -> list[str]:
        """Greedy completions (GSM8K and chat answers) via llama-server."""
        raise NotImplementedError
