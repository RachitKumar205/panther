"""Benchmark questions: one record per question from each source's rows; how each is asked (lm-evaluation-harness
conventions); and the files llama-perplexity reads, each checked by reading it back as llama.cpp does.

A record: {"id": "<task>/<split>/<n>", "task", "key": a hash of its normalized text, ...the question}. Two records with
the same key are the same question, whatever split they come from."""

from __future__ import annotations

import hashlib
import io
import re
import struct
import unicodedata
from collections import defaultdict

LETTERS = ("A", "B", "C", "D")


def normalize(s: str) -> str:
    """For comparing texts: NFKC, lower case, every run of non-alphanumerics one space."""
    return re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", s).lower()).strip()


def words(s: str) -> list[str]:
    return normalize(s).split()


def record(task: str, split: str, n, key_text: str, **fields) -> dict:
    return {"id": f"{task}/{split}/{n}", "task": task,
            "key": hashlib.sha256(normalize(key_text).encode()).hexdigest()[:16], **fields}


# --- records from each source's rows ---

def mmlu(rows: list[dict], split: str) -> list[dict]:
    return [record("mmlu", split, i, r["question"] + " || " + " | ".join(r["choices"]), subject=r["subject"],
                   question=r["question"], choices=list(r["choices"]), label=int(r["answer"]))
            for i, r in enumerate(rows)]


def hellaswag(rows: list[dict], split: str) -> list[dict]:
    # keyed on the context alone: HellaSwag asks several questions per context
    return [record("hellaswag", split, r["ind"], r["ctx"], activity_label=r["activity_label"], ctx_a=r["ctx_a"],
                   ctx_b=r["ctx_b"], endings=list(r["endings"]), label=int(r["label"])) for r in rows]


def winogrande(rows: list[dict], split: str) -> list[dict]:
    return [record("winogrande", split, i, r["sentence"], sentence=r["sentence"], option1=r["option1"],
                   option2=r["option2"], label=int(r["answer"]) - 1) for i, r in enumerate(rows)]


def arc_challenge(rows: list[dict], split: str) -> list[dict]:
    return [record("arc_challenge", split, r["id"], r["question"], question=r["question"],
                   choices=list(r["choices"]["text"]), label=list(r["choices"]["label"]).index(r["answerKey"]))
            for r in rows]


def gsm8k(rows: list[dict], split: str) -> list[dict]:
    return [record("gsm8k", split, i, r["question"], question=r["question"], answer=r["answer"])
            for i, r in enumerate(rows)]


RECORDS = {"mmlu": mmlu, "hellaswag": hellaswag, "winogrande": winogrande, "arc_challenge": arc_challenge,
           "gsm8k": gsm8k}


def text_of(r: dict) -> str:
    """Everything a model reads of a question (for the chat prompts' overlap check)."""
    skip = {"id", "task", "key", "subject", "label"}
    return "\n".join(x for k, v in r.items() if k not in skip for x in (v if isinstance(v, list) else [v]))


# --- how a question is asked: (question text, answer texts, index of the right answer) ---

def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def _mmlu_block(r: dict) -> str:
    choices = [f"{a}. {c}" for a, c in zip(LETTERS, r["choices"], strict=True)]
    return "\n".join([r["question"].strip(), *choices]) + "\nAnswer:"


def mmlu_shots(dev: list[dict]) -> dict[str, list[dict]]:
    """MMLU's 5 worked examples per subject."""
    by = defaultdict(list)
    for r in dev:
        by[r["subject"]].append(r)
    assert all(len(v) == 5 for v in by.values()), "MMLU dev should hold 5 examples per subject"
    return dict(by)


def asked(r: dict, shots: dict[str, list[dict]]) -> tuple[str, list[str], int]:
    if r["task"] == "hellaswag":
        q = _hellaswag_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        return q, [_hellaswag_clean(e) for e in r["endings"]], r["label"]
    if r["task"] == "arc_challenge":
        return f"Question: {r['question']}\nAnswer:", list(r["choices"]), r["label"]
    head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
    examples = "".join(f"{_mmlu_block(x)} {LETTERS[x['label']]}\n\n" for x in shots[r["subject"]])
    return head + examples + _mmlu_block(r), list(LETTERS), r["label"]


# --- llama.cpp's binary multiple-choice file (perplexity.cpp, multiple_choice_task::deserialize) ---

def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def write_choice(records: list[dict], shots: dict[str, list[dict]]) -> bytes:
    """A count, an offset per question, then each question: its text, its answers, which is right, no second set."""
    blobs = []
    for r in records:
        q, answers, right = asked(r, shots)
        assert q and answers and all(answers), f"{r['id']}: llama.cpp cannot read an empty question or answer"
        blobs.append(_string(q) + struct.pack("<I", len(answers)) + b"".join(map(_string, answers))
                     + struct.pack(f"<{len(answers)}i", *[int(i == right) for i in range(len(answers))])
                     + struct.pack("<I", 0))
    offsets, position = [], 4 + 4 * len(blobs)
    for b in blobs:
        offsets.append(position)
        position += len(b)
    return struct.pack("<I", len(blobs)) + struct.pack(f"<{len(blobs)}I", *offsets) + b"".join(blobs)


def read_choice(data: bytes) -> list[tuple[str, list[str], int]]:
    """As llama.cpp reads it: (question, answers, index of the right one) per question."""
    f = io.BytesIO(data)

    def u32() -> int:
        return struct.unpack("<I", f.read(4))[0]

    def string() -> str:
        return f.read(u32()).decode("utf-8")
    out = []
    for i, offset in enumerate([u32() for _ in range(u32())]):
        assert f.tell() == offset, f"question {i} at {f.tell()}, its offset says {offset}"
        q, n = string(), u32()
        answers = [string() for _ in range(n)]
        labels = struct.unpack(f"<{n}i", f.read(4 * n))
        assert u32() == 0 and sum(labels) == 1
        out.append((q, answers, labels.index(1)))
    assert not f.read(), "trailing bytes"
    return out


# --- WinoGrande: the CSV llama-perplexity --winogrande reads (load_winogrande_from_csv) ---

def wino_writable(r: dict) -> bool:
    """llama.cpp's CSV reader has no escaping: an option with a comma, any double quote, or not exactly one _ cannot
    be written."""
    s = r["sentence"]
    return ("," not in r["option1"] + r["option2"] and '"' not in s + r["option1"] + r["option2"]
            and s.count("_") == 1 and "\n" not in s)


def write_wino(records: list[dict]) -> bytes:
    rows = []
    for i, r in enumerate(records):
        assert wino_writable(r), f"{r['id']}: llama.cpp's CSV reader cannot read this question"
        s = f'"{r["sentence"]}"' if "," in r["sentence"] else r["sentence"]
        rows.append(f"{i},{s},{r['option1']},{r['option2']},{r['label'] + 1}")
    return ("\n".join(rows) + "\n\n").encode("utf-8")  # -f strips one newline; a last line at EOF is dropped


def read_wino(data: bytes) -> list[tuple[str, str, str, str, int]]:
    """As llama-perplexity reads `-f file.csv --winogrande`: (before _, after _, option 1, option 2, answer 1|2)."""
    text = data.decode("utf-8")
    text = text[:-1] if text.endswith("\n") else text
    *lines, last = text.split("\n")
    assert last == "", "a last line without a newline would be dropped"
    out = []
    for line in lines:
        commas, quoted = [], False
        for i, ch in enumerate(line):  # the first four commas outside double quotes
            if ch == '"':
                quoted = not quoted
            elif ch == "," and not quoted:
                commas.append(i)
                if len(commas) == 4:
                    break
        assert len(commas) == 4, f"llama.cpp would drop this line: {line[:60]!r}"
        c0, c1, c2, c3 = commas
        sentence = line[c0 + 2:c1 - 1] if line[c0 + 1] == '"' else line[c0 + 1:c1]
        before, after = sentence.split("_")
        out.append((before, after, line[c1 + 1:c2], line[c2 + 1:c3], int(line[c3 + 1:])))
    return out
