"""The dataset (README "The dataset"): train / select / test, each a web part and a chat part (the original's answers);
the chat prompts, the split check (decision 3) and the select text."""

from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from panther.model import DATA, lock, note, sha

SPLITS = ("train", "select", "test")
TEXT = DATA / "splits/text"
WEB = dict(zip(SPLITS, (DATA / "splits/calibration.txt", TEXT / "selection.txt", TEXT / "test_web.txt")))
# Every stage reads the whole of its split. select is read by every genome, so it is kept small: the chat part and
# whole web documents in a fixed random order up to SELECT_TOKENS in all. Its name carries the definition, so a
# select text, reference and cached KL records from another definition are never reused
SELECT_TOKENS = 49 * 2048
REF = {"select": "select100k", "test": "test"}  # each scored split's joined text and .kld reference
# The chat prompts: English first messages of OpenAssistant oasst1 (Apache-2.0), a fixed draw, none overlapping a
# benchmark question
PROMPTS = {"repo": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
           "file": "data/train-00000-of-00001-b42a775f407cee45.parquet",
           "n": {"train": 300, "select": 150, "test": 150}}
PROMPT_FILE = TEXT / "prompts.json"
LABELS = ("id", "task", "source_split", "key", "subject")  # a record's bookkeeping, not its question


def chat_text(model: str, split: str) -> Path:
    """A split's chat part: the original's own answers to that split's prompts, in its chat template (made by refs)."""
    return TEXT / f"chat-{model}-{split}.txt"


def joined_text(model: str, split: str) -> Path:
    """A split's chat and web parts in one file (llama-perplexity reads one); its stem is the ppl run's text name."""
    return TEXT / f"{REF[split]}-{model}.txt"


def ids_of(web: Path) -> Path:
    return TEXT / f"{web.stem}.ids.json"


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def strings(x) -> list[str]:
    """Every string in a parsed JSON record."""
    vs = x.values() if isinstance(x, dict) else x if isinstance(x, list) else []
    return [x] if isinstance(x, str) else [y for v in vs for y in strings(v)]


def overlapping(prompts: list[str], texts: list[str], n: int = 8) -> set[int]:
    """The prompts that share a run of n words with any of the texts (one shorter than n words: found inside one),
    compared lowercased, on letters and digits only."""
    grams, shorts, hit = {}, {}, set()
    for k, p in enumerate(prompts):
        w = words(p)
        if len(w) < n:
            shorts[k] = f" {' '.join(w)} "
        for i in range(len(w) - n + 1):
            grams.setdefault(" ".join(w[i:i + n]), []).append(k)
    for t in texts:
        w = words(t)
        hit.update(k for i in range(len(w) - n + 1) for k in grams.get(" ".join(w[i:i + n]), ()))
        joined = f" {' '.join(w)} "
        hit.update(k for k, s in shorts.items() if s in joined)
    return hit


def jsonl(f: Path) -> list[dict]:  # not splitlines: questions hold U+2028
    return [json.loads(x) for x in f.read_text().split("\n") if x]


def questions(group: str) -> dict[str, list[list[str]]]:
    """A question split's records (tasks/<group>/*.jsonl), each as the strings of its question and answers, by file."""
    return {f.name: [strings({k: v for k, v in r.items() if k not in LABELS}) for r in jsonl(f)]
            for f in sorted((DATA / "splits/tasks" / group).glob("*.jsonl"))}


def benchmark() -> list[str]:
    """Every selection and held-out question, one text per file (with the held-out llama.cpp task files)."""
    qs = ["\n".join(y for r in rs for y in r) for g in ("selection", "test") for rs in questions(g).values()]
    return qs + [g.read_bytes().decode("utf-8", "ignore")
                 for g in (DATA / "splits/llama_tasks/poc_test").iterdir() if g.is_file()]


def prompts() -> None:
    """The chat prompts, once for every model: a fixed draw, deduplicated, none sharing 8 words with a benchmark
    question, cut into train / select / test. Needs the network."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    with lock("prompts"):
        if PROMPT_FILE.exists():
            return
        f = hf_hub_download(PROMPTS["repo"], PROMPTS["file"], repo_type="dataset", revision=PROMPTS["revision"])
        cols = ["text", "parent_id", "role", "lang", "deleted", "review_result"]
        t = pq.read_table(f, columns=cols).to_pydict()
        pool = {" ".join(words(x)): x for x, up, r, lang, gone, ok in zip(*(t[c] for c in cols))
                if up is None and r == "prompter" and lang == "en" and not gone and ok and 20 <= len(x) <= 1000}
        pool = sorted(pool.values())
        random.Random(0).shuffle(pool)
        n = sum(PROMPTS["n"].values())
        pool = pool[:2 * n]  # checked in this order; the first n clean ones are kept, then cut in that order
        bad = overlapping(pool, benchmark())
        keep = [x for k, x in enumerate(pool) if k not in bad][:n]
        if len(keep) != n:
            raise SystemExit(f"chat prompts: only {len(keep)} of {n} clean prompts")
        cut = [sp for sp, m in PROMPTS["n"].items() for _ in range(m)]
        PROMPT_FILE.write_text(json.dumps({**PROMPTS, "checked": len(pool), "dropped": len(bad), "prompts": [
            {"split": sp, "text": x} for sp, x in zip(cut, keep)]}, indent=1))
        note(f"chat prompts: {n} from {PROMPTS['repo']} ({PROMPTS['n']}); {len(bad)} of {len(pool)} dropped as "
             f"overlapping a benchmark question")


def check() -> None:
    """Decision 3: the splits are their manifests' files and share no document, question or prompt; the files measured
    hold exactly the checked questions; no prompt shares 8 words with a benchmark question. Else the run stops."""
    s, bad = DATA / "splits", []
    want = json.loads((s / "manifest.json").read_text())["files"]
    tasks = json.loads((s / "llama_tasks/manifest.json").read_text())["files"]
    want |= {e["file"]: e["sha256"] for e in tasks.values()}
    used = [*WEB.values(), *(s / "tasks/selection").glob("*.jsonl"), *(s / "tasks/test").glob("*.jsonl")]
    bad += [f"{f.relative_to(s)}: not in the manifest" for f in used if str(f.relative_to(s)) not in want]
    bad += [f"{f}: missing" if not (s / f).exists() else f"{f}: hash differs" for f, h in want.items()
            if not (s / f).exists() or sha(s / f) != h]
    for t in ("mmlu", "hellaswag", "arc_challenge", "winogrande"):  # llama.cpp's files: their questions' ids
        sel, held = ({r["id"] for r in jsonl(s / f"tasks/{sp}/{t}.jsonl")} for sp in ("selection", "test"))
        if set(tasks[f"selection/{t}"]["ids"]) != sel or not set(tasks[f"poc_test/{t}"]["ids"]) <= held - sel:
            bad.append(f"llama_tasks {t}: not the selection questions, or held-out ones not only from the test split")
    test = {json.dumps(r, sort_keys=True) for r in jsonl(s / "tasks/test/gsm8k.jsonl")}
    if any(json.dumps(r, sort_keys=True) not in test for r in jsonl(s / "llama_tasks/poc_test/gsm8k.jsonl")):
        bad.append("llama_tasks/poc_test/gsm8k.jsonl: questions not in the test split")
    ids = {sp: set(json.loads(ids_of(p).read_text())) for sp, p in WEB.items()}
    bad += [f"web {a} and {b}: {n} documents in both" for a, b in combinations(SPLITS, 2)
            if (n := len(ids[a] & ids[b]))]
    key = lambda r: " ".join(words(" ".join(r)))  # noqa: E731  (the whole question: stem, choices, answer)
    held = questions("test")
    for f, rs in questions("selection").items():
        if n := len({key(r) for r in rs} & {key(r) for r in held.get(f, [])}):
            bad.append(f"{f}: {n} selection questions also held out")
    ps = json.loads(PROMPT_FILE.read_text())["prompts"]
    if n := len(ps) - len({" ".join(words(x["text"])) for x in ps}):
        bad.append(f"chat: {n} prompts twice")
    if hit := overlapping([x["text"] for x in ps], benchmark()):
        bad.append(f"chat: {len(hit)} prompts share 8 words with a benchmark question")
    if bad:
        raise SystemExit("split check: " + "; ".join(bad))
    note(f"split check: {sum((s / f).exists() for f in want)} files match their hashes; web documents, questions and "
         f"chat prompts share nothing")


def select_text(model: str, bf16: Path) -> str:
    """The select split: the original's answers to the 150 select prompts and whole web documents of the web select
    text, up to SELECT_TOKENS tokens in all (counted with the model's tokenizer)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(bf16.parent), gguf_file=bf16.name)
    count = lambda x: len(tok(x + "\n\n", add_special_tokens=False).input_ids)  # noqa: E731
    docs, ids = WEB["select"].read_text().split("\n\n"), json.loads(ids_of(WEB["select"]).read_text())
    if len(docs) != len(ids):  # documents were joined by a blank line; one with a blank line inside splits in two
        note(f"select text: {len(docs)} pieces for {len(ids)} documents; pieces taken as documents")
    chat, web = chat_text(model, "select").read_text(), []
    total = count(chat)
    for i in np.random.default_rng(0).permutation(len(docs)):  # whole documents; the rest are used by nothing
        if (total := total + count(docs[i])) > SELECT_TOKENS:
            break
        web.append(docs[i])
    text = chat + "\n" + "\n\n".join(web)
    n = len(tok(text).input_ids)
    note(f"select text: {len(web)} of {len(docs)} web documents, {n} tokens, {n // 2048} chunks read "
         f"({n % 2048} tokens in the last partial chunk not read)")
    return text

