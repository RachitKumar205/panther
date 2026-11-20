"""The dataset: sources, the train/select/test split and the leakage checks.

First component (`docs/algorithms/01-dataset-splits.md`). Two split dimensions, each pairwise
disjoint: text (train/select/test) and benchmark questions (selection/held-out). A prompt is
dropped if it shares ``n`` consecutive words with any benchmark question.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SplitName(StrEnum):
    """A text split."""

    TRAIN = "train"
    SELECT = "select"
    TEST = "test"


def words(text: str) -> list[str]:
    """Lowercase alphanumeric runs — the token unit of the overlap check."""
    return re.findall(r"[a-z0-9]+", text.lower())


class ChatPrompt(BaseModel):
    """One chat prompt, assigned to a split."""

    model_config = ConfigDict(frozen=True)

    text: str  # the prompt
    split: SplitName  # assigned split


class Question(BaseModel):
    """One benchmark question."""

    model_config = ConfigDict(frozen=True)

    id: str  # stable id within its task
    task: str  # mmlu, arc_challenge, hellaswag, winogrande, gsm8k
    stem: str  # the question
    choices: tuple[str, ...] = ()  # answer choices (empty for generative tasks)
    answer: str  # correct choice letter, or the numeric answer for GSM8K

    def key(self) -> str:
        """Canonical key for dedup and question-level overlap (stem + choices + answer)."""
        return " ".join(words(" ".join((self.stem, *self.choices, self.answer))))


class WebDocument(BaseModel):
    """One web document; cross-split disjointness is by ``id``."""

    model_config = ConfigDict(frozen=True)

    id: str  # dataset document id
    text: str  # document body


class SplitSpec(BaseModel):
    """The split's sizes and seed."""

    model_config = ConfigDict(frozen=True)

    prompt_counts: dict[SplitName, int] = Field(
        default_factory=lambda: {SplitName.TRAIN: 300, SplitName.SELECT: 150, SplitName.TEST: 150}
    )  # chat prompts per text split
    web_tokens: dict[SplitName, int] = Field(
        default_factory=lambda: {SplitName.TRAIN: 300_000, SplitName.SELECT: 100_352, SplitName.TEST: 150_000}
    )  # web token budget per text split
    selection_questions: dict[str, int] = Field(
        default_factory=lambda: {"mmlu": 300, "arc_challenge": 200, "hellaswag": 200, "winogrande": 200}
    )  # questions kept for selection, per benchmark
    seed: int = 0  # every draw is reproducible
    overlap_n: int = 8  # prompts sharing this many words with a question are dropped


def overlapping(prompts: Sequence[str], texts: Sequence[str], n: int = 8) -> frozenset[int]:
    """Indices of prompts that share a run of ``n`` words with any text.

    Builds a map from each prompt's n-grams to its index, then scans each text's n-grams. A prompt
    shorter than ``n`` words is flagged if its whole token sequence appears inside a text.
    """
    grams: dict[str, list[int]] = {}
    short: dict[int, str] = {}
    for i, prompt in enumerate(prompts):
        toks = words(prompt)
        if len(toks) < n:
            short[i] = f" {' '.join(toks)} "
        for j in range(len(toks) - n + 1):
            grams.setdefault(" ".join(toks[j : j + n]), []).append(i)
    hits: set[int] = set()
    for text in texts:
        toks = words(text)
        for j in range(len(toks) - n + 1):
            hits.update(grams.get(" ".join(toks[j : j + n]), ()))
        joined = f" {' '.join(toks)} "
        hits.update(i for i, s in short.items() if s in joined)
    return frozenset(hits)


def split_prompts(
    prompts: Sequence[str],
    counts: Mapping[SplitName, int],
    seed: int = 0,
) -> dict[SplitName, list[str]]:
    """Shuffle deterministically, then cut into the given counts in train/select/test order."""
    order = random.Random(seed).sample(range(len(prompts)), len(prompts))
    shuffled = [prompts[i] for i in order]
    out: dict[SplitName, list[str]] = {}
    start = 0
    for name in (SplitName.TRAIN, SplitName.SELECT, SplitName.TEST):
        take = counts[name]
        out[name] = shuffled[start : start + take]
        start += take
    return out


def partition_questions(
    questions: Mapping[str, Sequence[str]],
    selection: Mapping[str, int],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split each task's questions into selection (first ``k``) and held-out (the rest)."""
    sel: dict[str, list[str]] = {}
    held: dict[str, list[str]] = {}
    for task, keys in questions.items():
        k = selection[task]
        sel[task] = list(keys[:k])
        held[task] = list(keys[k:])
    return sel, held


def fill_to_budget(
    docs: Sequence[str],
    budget: int,
    count_tokens: Callable[[str], int],
    seed: int = 0,
) -> list[str]:
    """Whole documents in a seeded order until the next would exceed ``budget`` tokens."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    out: list[str] = []
    used = 0
    for i in order:
        n = count_tokens(docs[i])
        if used + n > budget:
            break
        out.append(docs[i])
        used += n
    return out
