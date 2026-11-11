"""GSM8K, reproducing lm-evaluation-harness 0.4.13's `gsm8k` task (D41): prompt layout, stop strings, the
strict-match and flexible-extract filters, and exact-match normalisation, copied from its gsm8k.yaml."""

from __future__ import annotations

import re

STOP = ["Question:", "</s>", "<|im_end|>"]
MAX_TOKENS = 256

STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]
INVALID = "[invalid]"  # lm-eval's fallback when a filter finds nothing


def prompt(question: str, fewshot: list[dict]) -> str:
    """5-shot: "Question: q\\nAnswer: a" exemplars joined by a blank line, then the question."""
    shots = "\n\n".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in fewshot)
    return f"{shots}\n\nQuestion: {question}\nAnswer:"


def strict_match(completion: str) -> str:
    m = STRICT.search(completion)
    return m.group(1) if m else INVALID


def flexible_extract(completion: str) -> str:
    matches = FLEXIBLE.findall(completion)
    if not matches:
        return INVALID
    last = matches[-1]  # group_select: -1
    return next((g for g in last if g), INVALID)


def normalise(s: str) -> str:
    for rx in IGNORE:
        s = re.sub(rx, "", s)
    return s.strip().lower()


def exact_match(prediction: str, reference_answer: str) -> bool:
    return normalise(prediction) == normalise(reference_answer)
