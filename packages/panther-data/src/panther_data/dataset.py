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
from typing import Protocol

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
        k = selection.get(task, 0)  # a task absent from the spec is held out entirely
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


class DatasetSource(Protocol):
    """The raw sources; real implementations download, tests provide synthetic data."""

    def chat_prompts(self) -> Sequence[str]:
        """Ordered, filtered, deduplicated prompt texts."""
        raise NotImplementedError

    def questions(self) -> Mapping[str, Sequence[Question]]:
        """Ordered questions per benchmark task."""
        raise NotImplementedError

    def web_documents(self) -> Sequence[WebDocument]:
        """All web documents in one pool; the builder partitions them disjointly."""
        raise NotImplementedError


class BuiltDataset(BaseModel):
    """The built split: prompts, questions and web documents, per split."""

    model_config = ConfigDict(frozen=True)

    prompts: dict[SplitName, list[str]]  # split -> prompt texts
    selection: dict[str, list[str]]  # task -> question ids kept for selection
    heldout: dict[str, list[str]]  # task -> question ids held out
    web: dict[SplitName, list[WebDocument]]  # split -> documents (disjoint by id)


_TEXT_ORDER = (SplitName.TRAIN, SplitName.TEST, SplitName.SELECT)
"""Fill order for the web budgets: train, then test, then select."""


def partition_web(
    docs: Sequence[WebDocument],
    budgets: Mapping[SplitName, int],
    count_tokens: Callable[[str], int],
    seed: int = 0,
) -> dict[SplitName, list[WebDocument]]:
    """Assign whole documents, disjoint by id, filling train then test then select budgets.

    The select budget is the select-text cap; ``select_text`` later reconciles it with the chat
    answers (chat + web <= budget).
    """
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    out: dict[SplitName, list[WebDocument]] = {name: [] for name in _TEXT_ORDER}
    used: dict[SplitName, int] = {name: 0 for name in _TEXT_ORDER}
    for i in order:
        n = count_tokens(docs[i].text)
        for name in _TEXT_ORDER:
            if used[name] + n <= budgets[name]:
                out[name].append(docs[i])
                used[name] += n
                break
        else:
            break  # fits no remaining budget; the rest is unused
    return out


def build_splits(
    source: DatasetSource,
    spec: SplitSpec,
    count_tokens: Callable[[str], int],
) -> BuiltDataset:
    """Build the chat, question and web splits; a prompt overlapping any question is dropped."""
    questions = source.questions()
    texts = [" ".join((q.stem, *q.choices, q.answer)) for qs in questions.values() for q in qs]
    need = sum(spec.prompt_counts.values())
    candidates = list(source.chat_prompts())[: 2 * need]
    clean = [p for i, p in enumerate(candidates) if i not in overlapping(candidates, texts, spec.overlap_n)][:need]
    prompts = split_prompts(clean, spec.prompt_counts, spec.seed)
    selection, heldout = partition_questions(
        {task: [q.id for q in qs] for task, qs in questions.items()},
        spec.selection_questions,
    )
    web = partition_web(source.web_documents(), spec.web_tokens, count_tokens, spec.seed)
    return BuiltDataset(prompts=prompts, selection=selection, heldout=heldout, web=web)
