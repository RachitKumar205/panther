"""Benchmark items -> llama-perplexity input files (docs/PIPELINE.md A4, D25, D38).

HellaSwag, ARC-Challenge and MMLU (5-shot) are written in llama.cpp's binary multiple-choice format and run
with `--multiple-choice` (file order, so results map to question ids). WinoGrande is written as the CSV that
`--winogrande` reads. Prompts follow lm-evaluation-harness conventions.

Readers that mirror llama.cpp's parsing (`read_mc_binary`, `parse_winogrande_csv`) are used to verify every
file before it is evaluated.
"""

from __future__ import annotations

import io
import re
import struct
from collections import defaultdict
from dataclasses import dataclass

MC_TASKS = ("hellaswag", "arc_challenge", "mmlu")
LETTERS = ("A", "B", "C", "D")


# --- prompts (lm-evaluation-harness conventions) ----------------------------------------------------------

def hellaswag_preprocess(text: str) -> str:
    """lm-eval's HellaSwag clean-up."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_prompt(item: dict) -> tuple[str, list[str], int]:
    ctx = item["ctx_a"] + " " + item["ctx_b"].capitalize()
    query = hellaswag_preprocess(item["activity_label"] + ": " + ctx)
    return query, [hellaswag_preprocess(e) for e in item["endings"]], item["label"]


def arc_prompt(item: dict) -> tuple[str, list[str], int]:
    return f"Question: {item['question']}\nAnswer:", list(item["choices"]), item["label"]


def _mmlu_block(item: dict) -> str:
    lines = [item["question"].strip()] + [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])]
    return "\n".join(lines) + "\nAnswer:"


def mmlu_prompt(item: dict, fewshot: list[dict]) -> tuple[str, list[str], int]:
    subject = item["subject"].replace("_", " ")
    head = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    shots = "".join(f"{_mmlu_block(ex)} {LETTERS[ex['label']]}\n\n" for ex in fewshot)
    return head + shots + _mmlu_block(item), list(LETTERS), item["label"]


def mmlu_fewshot_by_subject(dev: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for ex in dev:
        by[ex["subject"]].append(ex)
    bad = {s: len(v) for s, v in by.items() if len(v) != 5}
    if bad:
        raise ValueError(f"MMLU dev should have 5 examples per subject: {bad}")
    return dict(by)


# --- binary multiple-choice format (llama.cpp perplexity.cpp, multiple_choice_task::deserialize) ----------

@dataclass(frozen=True)
class MCTask:
    question: str
    answers: list[str]
    labels: list[int]  # 1 for the correct answer, 0 otherwise


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _task_bytes(t: MCTask) -> bytes:
    if not t.question or not t.answers or any(not a for a in t.answers):
        raise ValueError("llama.cpp rejects empty questions or answers")
    if len(t.answers) > 100 or sum(t.labels) != 1:
        raise ValueError("expected <= 100 answers and exactly one correct")
    body = _string(t.question) + struct.pack("<I", len(t.answers))
    body += b"".join(_string(a) for a in t.answers)
    body += struct.pack(f"<{len(t.labels)}i", *t.labels)
    return body + struct.pack("<I", 0)  # mc2: no multiple-correct answers


def write_mc_binary(tasks: list[MCTask]) -> bytes:
    """uint32 n, uint32 offset[n] (absolute stream positions), then each task serialised."""
    blobs = [_task_bytes(t) for t in tasks]
    pos = 4 + 4 * len(blobs)
    offsets = []
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    return struct.pack("<I", len(blobs)) + struct.pack(f"<{len(blobs)}I", *offsets) + b"".join(blobs)


def read_mc_binary(data: bytes) -> list[MCTask]:
    """Mirror of llama.cpp's reader: the count, the offsets, then tasks read sequentially."""
    f = io.BytesIO(data)

    def u32() -> int:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError("truncated")
        return struct.unpack("<I", raw)[0]

    def string() -> str:
        n = u32()
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError("truncated string")
        return raw.decode("utf-8")

    def answers() -> tuple[list[str], list[int]]:
        n = u32()
        if n > 100:
            raise ValueError("more than 100 answers")
        ans = [string() for _ in range(n)]
        labels = list(struct.unpack(f"<{n}i", f.read(4 * n))) if n else []
        return ans, labels

    n_task = u32()
    offsets = [u32() for _ in range(n_task)]
    out = []
    for i in range(n_task):
        if f.tell() != offsets[i]:
            raise ValueError(f"task {i} at {f.tell()}, offset table says {offsets[i]}")
        q = string()
        a1, l1 = answers()
        a2, _ = answers()
        if a2:
            raise ValueError("unexpected mc2 answers")
        out.append(MCTask(q, a1, l1))
    if f.read():
        raise ValueError("trailing bytes")
    return out


def mc_tasks(task: str, items: list[dict], fewshot: dict[str, list[dict]] | None = None) -> list[MCTask]:
    out = []
    for it in items:
        if task == "hellaswag":
            q, ans, gold = hellaswag_prompt(it)
        elif task == "arc_challenge":
            q, ans, gold = arc_prompt(it)
        elif task == "mmlu":
            assert fewshot is not None
            q, ans, gold = mmlu_prompt(it, fewshot[it["subject"]])
        else:
            raise ValueError(task)
        out.append(MCTask(q, ans, [int(i == gold) for i in range(len(ans))]))
    return out


# --- WinoGrande CSV (llama.cpp load_winogrande_from_csv) --------------------------------------------------

def write_winogrande_csv(items: list[dict]) -> str:
    rows = []
    for i, it in enumerate(items):
        sentence = it["sentence"]
        if "," in it["option1"] or "," in it["option2"] or '"' in it["option1"] + it["option2"] + sentence:
            raise ValueError(f"{it['id']}: llama.cpp's CSV parser cannot represent this row")
        if sentence.count("_") != 1 or "\n" in sentence:
            raise ValueError(f"{it['id']}: needs exactly one '_' and no newline")
        field = f'"{sentence}"' if "," in sentence else sentence
        rows.append(f"{i},{field},{it['option1']},{it['option2']},{it['label'] + 1}")
    # Two trailing newlines: llama.cpp's -f strips one, and its loader drops a last line that hits EOF.
    return "\n".join(rows) + "\n\n"


@dataclass(frozen=True)
class WinoRow:
    first: str
    second: str
    choices: tuple[str, str]
    answer: int  # 1 or 2


def parse_winogrande_csv(text: str) -> tuple[list[WinoRow], int]:
    """Port of what llama-perplexity does with `-f file.csv --winogrande`. Returns (rows kept, rows dropped).

    1. `-f` reads the file and strips one trailing newline.
    2. The loader reads lines with getline and stops when a read fails *or reaches EOF*, so a final line
       without a newline after it is silently discarded.
    3. Each line is split at its first four commas outside double quotes."""
    if text.endswith("\n"):
        text = text[:-1]
    pieces = text.split("\n")
    lines, dropped_at_eof = pieces[:-1], int(pieces[-1] != "")  # the last piece hit EOF
    out, skipped = [], dropped_at_eof
    for line in lines:
        comma, quote_open = [], False
        for i, ch in enumerate(line):
            if not quote_open:
                if ch == ",":
                    comma.append(i)
                    if len(comma) == 4:
                        break
                elif ch == '"':
                    quote_open = True
            elif ch == '"':
                quote_open = False
        if len(comma) != 4:
            skipped += 1
            continue
        c0, c1, c2, c3 = comma
        sentence = line[c0 + 2:c1 - 1] if line[c0 + 1] == '"' else line[c0 + 1:c1]
        choice1, choice2, answer = line[c1 + 1:c2], line[c2 + 1:c3], line[c3 + 1:]
        where = sentence.find("_")
        if where < 0 or answer.strip() not in ("1", "2"):
            skipped += 1
            continue
        out.append(WinoRow(sentence[:where], sentence[where + 1:], (choice1, choice2), int(answer)))
    return out, skipped
