"""Four disjoint data splits (PLAN.md §3): calibration, sensitivity, selection, test.

Pure functions only; downloading and file IO live in scripts/server/build_splits.py.

Web text: every FineWeb-Edu document is hashed (seeded) into one of four shards A-D, so the shards are
disjoint by construction. Each split takes whole documents from its own shard, in hash order, until its
token target is reached. Task items: test = full official sets; selection = seeded sample of non-test
splits; few-shot exemplars come from splits used by neither, and are excluded from selection.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

import numpy as np

WEB_SPLITS = ("calibration", "sensitivity", "selection", "test")  # shards A, B, C, D


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _key(seed: int, doc_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()[:8], "big")


def shard_of(seed: int, doc_id: str) -> int:
    return _key(seed, doc_id) % len(WEB_SPLITS)


@dataclass
class TextSplit:
    name: str
    doc_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    tokens: int = 0

    def joined(self) -> str:
        return "\n\n".join(self.texts)


def take_web_text(
    doc_ids: Sequence[str],
    get_text: Callable[[int], str],
    count_tokens: Callable[[str], int],
    targets: dict[str, int],
    seed: int,
    min_doc_tokens: int = 0,
    max_doc_tokens: int | None = None,
) -> dict[str, TextSplit]:
    """Fill each split from its own shard, documents in hash order, until its token target.

    doc_ids[i] identifies the document get_text(i) returns. Documents outside [min, max] tokens are
    skipped. Raises if a shard runs out before its target.
    """
    keys = np.fromiter((_key(seed, d) for d in doc_ids), dtype=np.uint64, count=len(doc_ids))
    shards = keys % np.uint64(len(WEB_SPLITS))
    out: dict[str, TextSplit] = {}
    for s, name in enumerate(WEB_SPLITS):
        target = targets[name]
        split = TextSplit(name)
        idx = np.flatnonzero(shards == s)
        for i in idx[np.argsort(keys[idx], kind="stable")]:
            if split.tokens >= target:
                break
            text = get_text(int(i))
            n = count_tokens(text)
            if n < min_doc_tokens or (max_doc_tokens is not None and n > max_doc_tokens):
                continue
            split.doc_ids.append(doc_ids[int(i)])
            split.texts.append(text)
            split.tokens += n
        if split.tokens < target:
            raise ValueError(f"shard for {name} exhausted at {split.tokens} < {target} tokens")
        out[name] = split
    return out


# --- task items -------------------------------------------------------------------------------

def norm(s: str) -> str:
    """Normalisation for overlap checks: NFKC, lower case, collapse non-alphanumerics."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z]+", " ", s).strip()


def _item(task: str, split: str, local_id: Any, key_text: str, **fields: Any) -> dict[str, Any]:
    return {"id": f"{task}/{split}/{local_id}", "task": task, "source_split": split,
            "key": sha256(norm(key_text)), **fields}


def hellaswag_items(rows: Iterable[dict], split: str) -> list[dict]:
    return [
        _item("hellaswag", split, r["ind"], r["ctx"],
              activity_label=r["activity_label"], ctx_a=r["ctx_a"], ctx_b=r["ctx_b"], ctx=r["ctx"],
              endings=list(r["endings"]), label=int(r["label"]))
        for r in rows
    ]


def winogrande_items(rows: Iterable[dict], split: str) -> list[dict]:
    return [
        _item("winogrande", split, i, r["sentence"],
              sentence=r["sentence"], option1=r["option1"], option2=r["option2"], label=int(r["answer"]) - 1)
        for i, r in enumerate(rows)
    ]


def arc_items(rows: Iterable[dict], split: str) -> list[dict]:
    out = []
    for r in rows:
        labels = list(r["choices"]["label"])
        out.append(_item("arc_challenge", split, r["id"], r["question"],
                         question=r["question"], choices=list(r["choices"]["text"]),
                         choice_labels=labels, label=labels.index(r["answerKey"])))
    return out


def mmlu_items(rows: Iterable[dict], split: str) -> list[dict]:
    return [
        _item("mmlu", split, f"{r['subject']}/{i}", r["question"] + " || " + " | ".join(r["choices"]),
              subject=r["subject"], question=r["question"], choices=list(r["choices"]), label=int(r["answer"]))
        for i, r in enumerate(rows)
    ]


def gsm8k_items(rows: Iterable[dict], split: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(_item("gsm8k", split, i, r["question"],
                         question=r["question"], answer=r["answer"], final=final))
    return out


ITEM_BUILDERS: dict[str, Callable[[Iterable[dict], str], list[dict]]] = {
    "hellaswag": hellaswag_items,
    "winogrande": winogrande_items,
    "arc_challenge": arc_items,
    "mmlu": mmlu_items,
    "gsm8k": gsm8k_items,
}


def sample(items: Sequence[dict], n: int | str, seed: int, exclude_keys: AbstractSet[str] = frozenset(),
           exclude_ids: AbstractSet[str] = frozenset()) -> tuple[list[dict], int]:
    """Seeded sample of n items (or all), skipping excluded content keys and ids.

    Returns (sample in original order, number of items dropped as overlapping)."""
    pool = [it for it in items if it["id"] not in exclude_ids]
    kept = [it for it in pool if it["key"] not in exclude_keys]
    dropped = len(pool) - len(kept)
    if n == "all":
        return kept, dropped
    assert isinstance(n, int)
    if n > len(kept):
        raise ValueError(f"asked for {n} items, only {len(kept)} available")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(kept), size=n, replace=False))
    return [kept[i] for i in pick], dropped


def check_disjoint(groups: dict[str, Iterable[dict]]) -> None:
    """Raise if any id or content key appears in more than one group."""
    seen_id: dict[str, str] = {}
    seen_key: dict[str, str] = {}
    for g, items in groups.items():
        for it in items:
            for seen, k in ((seen_id, it["id"]), (seen_key, it["key"])):
                if k in seen and seen[k] != g:
                    raise ValueError(f"{it['id']} overlaps between {seen[k]} and {g}")
                seen[k] = g
