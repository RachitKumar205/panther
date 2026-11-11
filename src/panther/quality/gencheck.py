"""Generation sanity (PLAN §2.1, D26): 50 fixed prompts, greedy, 128 tokens, compared with BF16."""

from __future__ import annotations

from collections import Counter

import numpy as np

PER_TASK = 10
SEED = 20261111
MAX_TOKENS = 128
LETTERS = "ABCDE"


def _options(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def make_prompts(selection: dict[str, list[dict]]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for task in ("hellaswag", "arc_challenge", "mmlu", "winogrande", "gsm8k"):
        items = selection[task]
        for i in sorted(rng.choice(len(items), size=PER_TASK, replace=False).tolist()):
            it = items[i]
            if task == "hellaswag":
                text = f"Continue this text: {it['ctx']}"
            elif task in ("arc_challenge", "mmlu"):
                text = (f"{it['question']}\n{_options(it['choices'])}\n"
                        "Answer with the letter and one sentence of reasoning.")
            elif task == "winogrande":
                text = f"Fill in the blank: {it['sentence']}"
            else:
                text = it["question"]
            out.append({"id": it["id"], "task": task, "prompt": text})
    return out


def matched_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def degenerate(text: str) -> bool:
    """Empty, or more than half of its word 4-grams are repeats."""
    words = text.split()
    if not words:
        return True
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if len(grams) < 8:
        return False
    repeats = sum(c - 1 for c in Counter(grams).values())
    return repeats / len(grams) > 0.5
