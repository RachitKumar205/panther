"""The dataset: four sets that never share a web document, a chat prompt or a question (GA, winner, distill,
verdict), plus QAT's documents, in none of them. Built once from pinned sources; each model's own chat answers and
text files are made from it."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from panther.data.questions import (
    RECORDS,
    mmlu_shots,
    normalize,
    read_choice,
    read_wino,
    text_of,
    wino_writable,
    words,
    write_choice,
    write_wino,
)
from panther.fitness.resources import Resources
from panther.model import Model
from panther.model.model import DATA, REPO
from panther.verdict.gsm8k import LlamaServer

SETS = ("distill", "ga", "winner", "verdict")  # the four sets; QAT's documents are a fifth web shard
SCORED = ("ga", "winner", "verdict")  # the sets with benchmark questions
TASKS = ("mmlu", "hellaswag", "arc_challenge", "winogrande")  # llama-perplexity's tasks (GSM8K: the verdict only)
LEAK_WORDS, SHORT_STEM = 12, 8  # a document leaks a question if it holds 12 of its words in a row, or the whole stem
# of a question of 8 to 11 words; stems under 8 words are too common to mean a leak ("Which of these is true?")


@dataclass(frozen=True)
class Set:
    """One scored set for one model: its text (the model's chat answers, then the web text) and its questions."""

    text: Path
    questions: dict[str, Path]  # task → llama.cpp file


@dataclass(frozen=True)
class ModelSets:
    distill: Path
    ga: Set
    winner: Set
    verdict: Set
    gsm8k_questions: Path
    gsm8k_examples: Path
    qat_tokens: np.ndarray
    qat_validation: np.ndarray


class Hub:
    """The pinned sources, from Hugging Face. Records each downloaded file's hash for the manifest."""

    def __init__(self, config: dict) -> None:
        self.config, self.files = config, {}

    def download(self, repo: str, revision: str, name: str) -> Path:
        path = Path(hf_hub_download(repo, name, revision=revision, repo_type="dataset"))
        with open(path, "rb") as f:
            self.files[f"{repo}/{name}@{revision}"] = hashlib.file_digest(f, "sha256").hexdigest()
        return path

    def count(self):
        """Tokens of a text, by the pinned tokenizer."""
        t = self.config["tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(t["repo"], revision=t["revision"])
        return lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)

    def web(self) -> tuple[list[str], list[str]]:
        w = self.config["web"]
        tables = [pq.read_table(self.download(w["repo"], w["revision"], f), columns=["id", "text"]) for f in w["files"]]
        return ([i for t in tables for i in t.column("id").to_pylist()],
                [x for t in tables for x in t.column("text").to_pylist()])

    def rows(self, task: str, split: str) -> list[dict]:
        b = self.config["benchmarks"][task]
        return pq.read_table(self.download(b["repo"], b["revision"], b["files"][split])).to_pylist()

    def prompts(self) -> list[dict]:
        c = self.config["chat"]
        return pq.read_table(self.download(c["repo"], c["revision"], c["file"])).to_pylist()


class Dataset:
    """Everything under data/dataset/<digest of configs/dataset.yaml>/."""

    def __init__(self, config: Path = REPO / "configs/dataset.yaml") -> None:
        self.config = yaml.safe_load(config.read_text())
        self.seed = self.config["seed"]
        self.dir = DATA / "dataset" / hashlib.sha256(config.read_bytes()).hexdigest()[:12]
        self.web_distill = self.dir / "web" / "distill.txt"

    # --- the shared sets (network, no GPU) ---

    def build(self, sources: Hub | None = None) -> None:
        """Downloads the pinned sources and writes the shared sets; skipped if done. Written in a .part folder,
        renamed when complete."""
        if (self.dir / "manifest.json").exists():
            return
        sources = sources or Hub(self.config)
        part = self.dir.with_name(self.dir.name + ".part")
        if part.exists():  # a stopped build's leftovers
            shutil.rmtree(part)
        manifest = {"seed": self.seed}

        questions, examples, every = self.questions(sources, manifest)
        manifest["web"] = self.web(sources, part / "web", every)
        prompts = self.prompts(sources, every, manifest)
        for s, texts in prompts.items():
            self.write(part / "prompts" / f"{s}.json", json.dumps(texts, indent=1))
        shots = mmlu_shots(examples["mmlu"])
        for s in SCORED:
            for task in TASKS:
                records = questions[s][task]
                self.write(part / "questions" / s / f"{task}.jsonl", "".join(json.dumps(r) + "\n" for r in records))
                if task == "winogrande":
                    data = write_wino(records)
                    assert read_wino(data) == [(*r["sentence"].split("_"), r["option1"], r["option2"], r["label"] + 1)
                                               for r in records], f"{s}/{task}: the file does not read back"
                    self.write(part / "questions" / s / "winogrande.csv", data)
                else:
                    data = write_choice(records, shots)
                    assert [x[2] for x in read_choice(data)] == [r["label"] for r in records], f"{s}/{task}: read back"
                    self.write(part / "questions" / s / f"{task}.bin", data)
        gsm8k = {"verdict": questions["verdict"]["gsm8k"], "examples": examples["gsm8k"]}
        for name, records in gsm8k.items():
            self.write(part / "questions" / f"gsm8k-{name}.jsonl",
                       "".join(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n" for r in records))
        self.check(questions, examples, prompts, manifest["web"])
        manifest["sources"] = sources.files
        self.write(part / "manifest.json", json.dumps(manifest, indent=1))
        part.rename(self.dir)

    def questions(self, sources: Hub, manifest: dict) -> tuple[dict, dict, list[dict]]:
        """The verdict's sample of each test split; the GA's and the winner's from the non-test splits (none in the
        full test split or the few-shot examples, WinoGrande's only those llama.cpp's CSV can hold), in a seeded
        shuffle. Returns {set: {task: records}}, {task: few-shot records}, and every record loaded."""
        questions = {s: {} for s in SCORED}
        examples, every, counts = {}, [], {}
        for i, (task, b) in enumerate(self.config["benchmarks"].items()):
            rng = random.Random(self.seed + i)
            build = RECORDS[task]
            test = build(sources.rows(task, b["verdict"]["split"]), b["verdict"]["split"])
            fewshot = build(sources.rows(task, b["fewshot"]["split"]), b["fewshot"]["split"]) if "fewshot" in b else []
            pool = [r for split in b.get("from", []) for r in build(sources.rows(task, split), split)]
            every += test + fewshot + pool
            dropped = {}
            if task == "winogrande":
                dropped["not writable as llama.cpp's CSV"] = sum(not wino_writable(r) for r in test + pool)
                test, pool = [r for r in test if wino_writable(r)], [r for r in pool if wino_writable(r)]
            questions["verdict"][task] = self.sample(test, b["verdict"]["n"], rng)
            if "fewshot" in b:
                n = b["fewshot"]["n"]
                examples[task] = fewshot if n == "all" else self.sample(
                    [r for r in fewshot if r["key"] not in {t["key"] for t in test}], n, rng)
            known = {r["key"] for r in test + examples.get(task, [])}
            unique = {r["key"]: r for r in pool if r["key"] not in known}  # one record per question
            dropped["in the test split or the few-shot examples, or twice"] = len(pool) - len(unique)
            pool = list(unique.values())
            rng.shuffle(pool)
            if "ga" in b:
                assert len(pool) >= b["ga"], f"{task}: {len(pool)} selection questions, the GA set needs {b['ga']}"
                questions["ga"][task] = pool[:b["ga"]]
                questions["winner"][task] = pool[b["ga"]:b["ga"] + b["winner"]]
            counts[task] = {s: len(questions[s].get(task, [])) for s in SCORED} | {"dropped": dropped}
        manifest["questions"] = counts
        return questions, examples, every

    def sample(self, records: list[dict], n, rng: random.Random) -> list[dict]:
        """n records ("all": every one), in a seeded shuffle."""
        records = list(records)
        rng.shuffle(records)
        return records if n == "all" else records[:n]

    def web(self, sources: Hub, folder: Path, every: list[dict]) -> dict:
        """FineWeb-Edu documents, each hashed with the seed into one of five shards (the four sets and QAT's): a
        document can be in one set only, and dropping one never moves another. A set takes its shard's documents in
        hash order, whole, until its token target, skipping documents outside the length range and documents that
        leak a benchmark question. Writes each set's text (documents joined by a blank line) and its documents' ids."""
        w = self.config["web"]
        count = sources.count()
        ids, texts = sources.web()
        names = [*SETS, "qat"]
        keys = np.array([int.from_bytes(hashlib.sha256(f"{self.seed}:{i}".encode()).digest()[:8], "big")
                         for i in ids], dtype=np.uint64)
        shard = keys % np.uint64(len(names))
        grams, stems = self.leak_index(every)
        report = {}
        for s, name in enumerate(names):
            members = np.flatnonzero(shard == s)
            taken, tokens, leaked, out_of_range = [], 0, 0, 0
            for i in members[np.argsort(keys[members], kind="stable")]:
                if tokens >= w["tokens"][name]:
                    break
                n = count(texts[i])
                if not w["doc_tokens"][0] <= n <= w["doc_tokens"][1]:
                    out_of_range += 1
                    continue
                if self.leaks(texts[i], grams, stems):
                    leaked += 1
                    continue
                taken.append(int(i))
                tokens += n
            assert tokens >= w["tokens"][name], f"web {name}: its shard ran out at {tokens} tokens"
            self.write(folder / f"{name}.txt", "\n\n".join(texts[i] for i in taken))
            self.write(folder / f"{name}.ids.json", json.dumps([ids[i] for i in taken]))
            report[name] = {"documents": len(taken), "tokens": tokens,
                            "dropped": {"leaking a benchmark question": leaked, "too short or too long": out_of_range}}
        return report

    def leak_index(self, records: list[dict]) -> tuple[set[str], list[str]]:
        """Every question's runs of 12 words, and the whole stems of questions of 8 to 11 words (made once)."""
        grams, stems = set(), []
        for r in records:
            stem = words(r.get("question") or r.get("ctx") or r.get("sentence") or "")
            if len(stem) >= LEAK_WORDS:
                grams.update(" ".join(stem[j:j + LEAK_WORDS]) for j in range(len(stem) - LEAK_WORDS + 1))
            elif len(stem) >= SHORT_STEM:
                stems.append(f" {' '.join(stem)} ")
        return grams, stems

    def leaks(self, text: str, grams: set[str], stems: list[str]) -> bool:
        w = words(text)
        if any(" ".join(w[j:j + LEAK_WORDS]) in grams for j in range(len(w) - LEAK_WORDS + 1)):
            return True
        joined = f" {' '.join(w)} "
        return any(s in joined for s in stems)

    def prompts(self, sources: Hub, every: list[dict], manifest: dict) -> dict[str, list[str]]:
        """English first prompter messages, not deleted, review passed, 20-1000 characters; deduplicated; any
        sharing 8 words in a row with a benchmark question dropped; a fixed shuffle cut into the sets."""
        c = self.config["chat"]
        lo, hi = c["chars"]
        pool = {normalize(r["text"]): r["text"] for r in sources.prompts()
                if r["parent_id"] is None and r["role"] == "prompter" and r["lang"] == "en" and not r["deleted"]
                and r["review_result"] and lo <= len(r["text"]) <= hi}
        pool = sorted(pool.values())
        random.Random(self.seed).shuffle(pool)
        need = sum(c["prompts"].values())
        candidates = pool[:2 * need]
        bad = self.overlapping(candidates, [text_of(r) for r in every], c["overlap_words"])
        clean = [p for i, p in enumerate(candidates) if i not in bad][:need]
        assert len(clean) == need, f"chat prompts: {len(clean)} clean, the sets need {need}"
        manifest["prompts"] = {"candidates": len(candidates), "dropped as sharing 8 words with a question": len(bad)}
        cut, out = 0, {}
        for s in SETS:
            out[s] = clean[cut:cut + c["prompts"][s]]
            cut += c["prompts"][s]
        return out

    def overlapping(self, prompts: list[str], texts: list[str], n: int) -> set[int]:
        """The prompts sharing n words in a row with any text (a prompt shorter than n words: found whole). The
        prompts' runs are indexed once, then every text is scanned once."""
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

    def check(self, questions: dict, examples: dict, prompts: dict, web: dict) -> None:
        """No question, prompt or web document in two sets; fails loudly."""
        for task in TASKS + ("gsm8k",):
            seen: dict[str, str] = {}
            groups = {s: questions[s].get(task, []) for s in SCORED} | {"fewshot": examples.get(task, [])}
            for s, records in groups.items():
                for r in records:
                    assert seen.get(r["key"], s) == s or s == "fewshot" and seen[r["key"]] == "verdict", \
                        f"{task}: {r['id']} in both {seen[r['key']]} and {s}"
                    seen.setdefault(r["key"], s)
        seen = {}
        for s, texts in prompts.items():
            for p in texts:
                assert seen.setdefault(normalize(p), s) == s, f"a chat prompt in both {seen[normalize(p)]} and {s}"
        ids: dict[str, str] = {}
        for name in web:
            for i in json.loads((self.dir.with_name(self.dir.name + ".part") / "web" / f"{name}.ids.json").read_text()):
                assert ids.setdefault(i, name) == name, f"web document {i} in both {ids[i]} and {name}"

    def write(self, path: Path, data: str | bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)

    # --- one model's sets (GPU, no network) ---

    def for_model(self, model: Model, qat_windows: int = 1024) -> ModelSets:
        """The model's own chat answers to every prompt (the original, greedy, in its chat template), each set's text
        (its chat answers, then its web text), and QAT's windows of the model's tokens. Made once, in
        data/models/<model>/sets/<dataset>/."""
        assert (self.dir / "manifest.json").exists(), f"{self.dir}: not built (Dataset().build() first)"
        out = model.dir / "sets" / self.dir.name
        answers = out / "chat.json"
        if not answers.exists():
            prompts = {s: json.loads((self.dir / "prompts" / f"{s}.json").read_text()) for s in SETS}
            flat = [p for s in SETS for p in prompts[s]]
            with LlamaServer(model.original, Resources(), chat=True) as server:
                said = iter(server.complete(flat))
            self.write(answers.with_suffix(".part"), json.dumps({s: [next(said) for _ in prompts[s]] for s in SETS}))
            os.replace(answers.with_suffix(".part"), answers)
        chat = json.loads(answers.read_text())
        for s in SETS:
            if not (out / f"{s}.txt").exists():
                web = (self.dir / "web" / f"{s}.txt").read_text()
                self.write(out / f"{s}.txt", "\n\n".join(chat[s]) + "\n\n" + web)

        q = self.config["qat"]
        windows = out / f"qat-{qat_windows}.npz"
        if not windows.exists():
            ids = model.tokens(self.dir / "web" / "qat.txt")
            need = (qat_windows + q["validation"]) * q["length"]
            assert len(ids) >= need, f"QAT: {len(ids)} tokens, {qat_windows} + {q['validation']} windows need {need}"
            cut = ids[:need].reshape(-1, q["length"])
            np.savez(windows.with_suffix(".part.npz"), tokens=cut[:qat_windows], validation=cut[qat_windows:])
            os.replace(windows.with_suffix(".part.npz"), windows)
        qat = np.load(windows)

        def scored(s: str) -> Set:
            folder = self.dir / "questions" / s
            return Set(out / f"{s}.txt", {t: folder / ("winogrande.csv" if t == "winogrande" else f"{t}.bin")
                                          for t in TASKS})
        gsm8k = self.dir / "questions"
        return ModelSets(out / "distill.txt", scored("ga"), scored("winner"), scored("verdict"),
                         gsm8k / "gsm8k-verdict.jsonl", gsm8k / "gsm8k-examples.jsonl",
                         qat["tokens"], qat["validation"])
