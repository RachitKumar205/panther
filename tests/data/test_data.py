"""The dataset, built from tiny synthetic sources (no network): the sets never share anything, every llama.cpp file
reads back as written, leaking documents and overlapping prompts are dropped; one model's sets on the tiny model."""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from panther.data import Dataset
from panther.data import dataset as dataset_module
from panther.data.questions import mmlu, mmlu_shots, read_choice, read_wino, write_choice, write_wino
from panther.model import Model
from panther.model import model as model_module
from tests.model.tiny import needs_perplexity, write_original

VOCABULARY = [f"w{i}" for i in range(3000)]


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(VOCABULARY) for _ in range(n))


LEAK = "these twelve words come from a benchmark question and must never appear in web text"


class FakeHub:
    """Synthetic sources shaped like the real ones. Every 10th web document holds 12 words of a benchmark question;
    one chat prompt shares 8 words with one."""

    files: dict = {}

    def __init__(self) -> None:
        self.rng = random.Random(0)

    def count(self):
        return lambda text: len(text.split())

    def web(self) -> tuple[list[str], list[str]]:
        texts = [sentence(self.rng, self.rng.randint(150, 400)) for _ in range(600)]
        for i in range(0, 600, 10):
            texts[i] += " " + LEAK
        return [f"d{i}" for i in range(600)], texts

    def rows(self, task: str, split: str) -> list[dict]:
        n = {"dev": 10, "validation": 40, "test": 60, "train": 50}[split]
        rng = random.Random(zlib.crc32(f"{task}/{split}".encode()))
        if task == "mmlu":
            rows = [{"question": sentence(rng, 15), "subject": f"s{i % 2}",
                     "choices": [sentence(rng, 2) for _ in "abcd"], "answer": rng.randrange(4)} for i in range(n)]
            if split == "validation":
                rows[0]["question"] = LEAK  # the question the planted documents leak
            return rows
        if task == "hellaswag":
            return [{"ind": i, "ctx": sentence(rng, 15), "ctx_a": sentence(rng, 8), "ctx_b": sentence(rng, 7),
                     "activity_label": "act", "endings": [sentence(rng, 4) for _ in range(4)],
                     "label": str(rng.randrange(4))} for i in range(n)]
        if task == "arc_challenge":
            return [{"id": f"{split}{i}", "question": sentence(rng, 15),
                     "choices": {"text": [sentence(rng, 2) for _ in range(4)], "label": list("ABCD")},
                     "answerKey": rng.choice("ABCD")} for i in range(n)]
        if task == "winogrande":
            rows = [{"sentence": sentence(rng, 6) + " _ " + sentence(rng, 6), "option1": "cat", "option2": "dog",
                     "answer": str(rng.choice((1, 2)))} for _ in range(n)]
            rows[0]["option1"] = "cat, dog"  # llama.cpp's CSV cannot hold this one
            return rows
        return [{"question": sentence(rng, 15), "answer": f"{i} + 1\n#### {i + 1}"} for i in range(n)]  # gsm8k

    def prompts(self) -> list[dict]:
        rows = [{"text": sentence(self.rng, 20), "parent_id": None, "role": "prompter", "lang": "en",
                 "deleted": False, "review_result": True} for _ in range(30)]  # all are candidates (2 x 15)
        rows[0]["text"] = "please tell me about " + " ".join(LEAK.split()[:8])  # shares 8 words with a question
        rows.append({**rows[1], "lang": "de"})  # not English: never used
        return rows


def small_config(path: Path) -> Path:
    c = yaml.safe_load((Path(__file__).parents[2] / "configs/dataset.yaml").read_text())
    c["web"]["doc_tokens"] = [100, 1000]
    c["web"]["tokens"] = {"distill": 3000, "ga": 1500, "winner": 1500, "verdict": 2000, "qat": 5000}
    c["chat"]["prompts"] = {"distill": 6, "ga": 3, "winner": 3, "verdict": 3}
    b = c["benchmarks"]
    b["mmlu"].update(verdict={"split": "test", "n": 20}, ga=8, winner=10)
    b["hellaswag"].update(verdict={"split": "validation", "n": 20}, ga=8, winner=10)
    b["arc_challenge"].update(ga=8, winner=10)
    b["winogrande"].update(ga=8, winner=10)
    b["gsm8k"].update(verdict={"split": "test", "n": 10})
    c["qat"] = {"length": 64, "validation": 4}
    path.write_text(yaml.safe_dump(c))
    return path


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dataset:
    for module in (model_module, dataset_module):
        monkeypatch.setattr(module, "DATA", tmp_path)
    ds = Dataset(small_config(tmp_path / "dataset.yaml"))
    ds.build(FakeHub())
    return ds


def records(ds: Dataset, s: str, task: str) -> list[dict]:
    return [json.loads(x) for x in (ds.dir / "questions" / s / f"{task}.jsonl").read_text().split("\n") if x]


# --- the shared sets ---

def test_the_sets_have_their_sizes_and_the_verdict_takes_every_question_of_a_small_test_split(built: Dataset) -> None:
    m = json.loads((built.dir / "manifest.json").read_text())["questions"]
    assert m["mmlu"]["ga"] == 8 and m["mmlu"]["winner"] == 10 and m["mmlu"]["verdict"] == 20
    assert m["arc_challenge"]["verdict"] == 60  # "all"
    assert m["winogrande"]["dropped"]["not writable as llama.cpp's CSV"] == 2  # one in train, one in validation
    assert len((built.dir / "questions/gsm8k-verdict.jsonl").read_text().split("\n")) - 1 == 10
    assert len((built.dir / "questions/gsm8k-examples.jsonl").read_text().split("\n")) - 1 == 5


def test_no_question_is_in_two_sets(built: Dataset) -> None:
    for task in ("mmlu", "hellaswag", "arc_challenge", "winogrande"):
        keys = [{r["key"] for r in records(built, s, task)} for s in ("ga", "winner", "verdict")]
        assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]), task


def test_web_documents_fill_their_sets_and_none_is_in_two(built: Dataset) -> None:
    web = json.loads((built.dir / "manifest.json").read_text())["web"]
    ids = {name: set(json.loads((built.dir / "web" / f"{name}.ids.json").read_text())) for name in web}
    names = list(ids)
    assert all(not ids[a] & ids[b] for a in names for b in names if a < b)
    for name, target in {"distill": 3000, "ga": 1500, "winner": 1500, "verdict": 2000, "qat": 5000}.items():
        assert web[name]["tokens"] >= target


def test_documents_leaking_a_benchmark_question_are_dropped(built: Dataset) -> None:
    web = json.loads((built.dir / "manifest.json").read_text())["web"]
    taken = {i for name in web for i in json.loads((built.dir / "web" / f"{name}.ids.json").read_text())}
    assert not taken & {f"d{i}" for i in range(0, 600, 10)}
    assert sum(w["dropped"]["leaking a benchmark question"] for w in web.values()) > 0
    assert LEAK not in (built.dir / "web" / "distill.txt").read_text()


def test_a_short_common_stem_does_not_drop_documents(built: Dataset) -> None:
    grams, stems = built.leak_index([{"question": "Which of these is true?"}, {"question": " ".join(VOCABULARY[:9])}])
    assert not built.leaks("many texts ask which of these is true? daily", grams, stems)  # under 8 words: skipped
    assert built.leaks("x " + " ".join(VOCABULARY[:9]) + " y", grams, stems)  # 9 words, whole: a leak


def test_dropping_a_leaking_document_moves_no_other_between_sets(built: Dataset, tmp_path: Path) -> None:
    def sets(every: list[dict]) -> dict[str, str]:
        built.web(FakeHub(), tmp_path / "web", every)
        return {i: name for name in ("distill", "ga", "winner", "verdict", "qat")
                for i in json.loads((tmp_path / "web" / f"{name}.ids.json").read_text())}
    kept, dropped = sets([]), sets([{"question": LEAK}])  # nothing leaks, then every 10th document leaks
    assert any(i not in dropped for i in kept)  # some documents were dropped
    assert all(kept[i] == name for i, name in dropped.items() if i in kept)  # none changed set


def test_chat_prompts_sharing_8_words_with_a_question_are_dropped_and_none_is_in_two_sets(built: Dataset) -> None:
    prompts = {s: json.loads((built.dir / "prompts" / f"{s}.json").read_text()) for s in ("distill", "ga", "winner",
                                                                                           "verdict")}
    every = [p for ps in prompts.values() for p in ps]
    assert len(every) == len(set(every)) == 15
    assert not any(p.startswith("please tell me about") for p in every)
    report = json.loads((built.dir / "manifest.json").read_text())["prompts"]
    assert report["dropped as sharing 8 words with a question"] == 1


def test_every_llama_cpp_file_reads_back_as_written(built: Dataset) -> None:
    for s in ("ga", "winner", "verdict"):
        for task in ("mmlu", "hellaswag", "arc_challenge"):
            answers = [x[2] for x in read_choice((built.dir / "questions" / s / f"{task}.bin").read_bytes())]
            assert answers == [r["label"] for r in records(built, s, task)]
        rows = read_wino((built.dir / "questions" / s / "winogrande.csv").read_bytes())
        assert [r[4] for r in rows] == [r["label"] + 1 for r in records(built, s, "winogrande")]


def test_the_check_catches_a_question_in_two_sets(built: Dataset) -> None:
    q = {s: {t: records(built, s, t) for t in ("mmlu", "hellaswag", "arc_challenge", "winogrande")}
         for s in ("ga", "winner", "verdict")}
    q["winner"]["mmlu"].append(q["ga"]["mmlu"][0])
    with pytest.raises(AssertionError, match="in both ga and winner"):
        built.check(q, {}, {}, {})


def test_a_changed_definition_is_a_new_dataset_and_a_built_one_is_not_rebuilt(built: Dataset, tmp_path: Path) -> None:
    c = yaml.safe_load((tmp_path / "dataset.yaml").read_text())
    c["seed"] += 1
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(c))
    assert Dataset(tmp_path / "other.yaml").dir != built.dir
    before = (built.dir / "manifest.json").stat().st_mtime_ns
    built.build(None)  # done: no sources needed
    assert (built.dir / "manifest.json").stat().st_mtime_ns == before


# --- the question formats ---

def test_the_mmlu_prompt_has_its_subjects_five_worked_examples() -> None:
    rows = [{"question": f"Q{i}?", "subject": "high_school_physics", "choices": ["a", "b", "c", "d"], "answer": i % 4}
            for i in range(6)]
    dev, item = mmlu(rows[:5], "dev"), mmlu(rows[5:], "test")
    data = write_choice(item, mmlu_shots(dev))
    question, answers, right = read_choice(data)[0]
    assert question.startswith("The following are multiple choice questions (with answers) about high school physics.")
    assert question.count("\nAnswer:") == 6 and question.endswith("Q5?\nA. a\nB. b\nC. c\nD. d\nAnswer:")
    assert answers == ["A", "B", "C", "D"] and right == 1


def test_winogrande_rows_round_trip_and_an_unwritable_one_is_refused() -> None:
    rows = [{"id": "w/0", "sentence": "The cat sat, then _ ran.", "option1": "it", "option2": "he", "label": 0}]
    assert read_wino(write_wino(rows)) == [("The cat sat, then ", " ran.", "it", "he", 1)]
    with pytest.raises(AssertionError):
        write_wino([{**rows[0], "option1": "a, b"}])


# --- one model's sets, on the tiny model ---

@needs_perplexity
def test_one_models_sets_hold_its_chat_answers_then_the_web_text(built: Dataset, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Model, "tokens", lambda self, text: np.arange(len(text.read_text().split()) * 2) % 200)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    write_original(m.original)
    sets = built.for_model(m, qat_windows=8)
    chat = json.loads((sets.distill.parent / "chat.json").read_text())
    assert [len(chat[s]) for s in ("distill", "ga", "winner", "verdict")] == [6, 3, 3, 3]
    text = sets.ga.text.read_text()
    assert text.startswith(chat["ga"][0]) and text.endswith((built.dir / "web/ga.txt").read_text())
    assert set(sets.ga.questions) == {"mmlu", "hellaswag", "arc_challenge", "winogrande"}
    assert all(f.exists() for f in sets.verdict.questions.values())
    assert sets.qat_tokens.shape == (8, 64) and sets.qat_validation.shape == (4, 64)
    before = sets.distill.stat().st_mtime_ns
    assert built.for_model(m, qat_windows=8).distill.stat().st_mtime_ns == before  # made once
