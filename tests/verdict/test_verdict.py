"""The winner, the verdict's statistics, GSM8K, and the verdict on the tiny model with the real llama.cpp."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from panther.fitness import Fitness, Score
from panther.fitness.resources import Resources
from panther.model import Model
from panther.model import model as model_module
from panther.presets import presets as presets_module
from panther.verdict import GSM8K, Verdict, Winner, judge
from panther.verdict import verdict as verdict_module
from panther.verdict import winner as winner_module
from panther.verdict.gsm8k import LlamaServer, correct
from panther.verdict.verdict import Measured
from tests.fitness.questions import ga_set
from tests.model.tiny import needs_perplexity, write_original

TASKS = ("mmlu", "hellaswag", "arc_challenge", "winogrande", "gsm8k")


# --- the statistics ---

def measured(rng: np.random.Generator, flip_rate: float, kl: float, original: dict, n_chunks: int = 40) -> Measured:
    """A file that flips each question with flip_rate and drifts kl per chunk (with some noise)."""
    right = {t: np.where(rng.random(len(o)) < flip_rate, ~o, o) for t, o in original.items()}
    return Measured(right, kl, 90.0, kl + 0.1 * kl * rng.standard_normal(n_chunks))


@pytest.fixture
def original() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {t: rng.random(2000) < 0.6 for t in TASKS}


def test_a_file_against_itself_is_100_and_a_tie(original) -> None:
    a = measured(np.random.default_rng(1), 0.1, 0.05, original)
    r = judge(a, a, original)
    assert r["relative"] == pytest.approx(100) and r["verdict"] == "TIE"
    assert r["interval"] == pytest.approx([100, 100])  # identical answers and chunks: no uncertainty


def test_clearly_better_wins_and_clearly_worse_loses(original) -> None:
    good = measured(np.random.default_rng(1), 0.05, 0.02, original)
    bad = measured(np.random.default_rng(2), 0.20, 0.10, original)
    assert judge(good, bad, original)["verdict"] == "WIN"
    assert judge(bad, good, original)["verdict"] == "LOSE"
    r = judge(good, bad, original)
    assert r["interval"][0] < r["relative"] < r["interval"][1] < 100


def test_a_small_difference_is_a_tie(original) -> None:
    a = measured(np.random.default_rng(1), 0.100, 0.050, original)
    b = measured(np.random.default_rng(2), 0.102, 0.051, original)
    assert judge(a, b, original)["verdict"] == "TIE"


def test_the_score_is_flips_plus_weight_times_kl_and_the_interval_is_1_96_standard_errors(original) -> None:
    a = measured(np.random.default_rng(1), 0.1, 0.05, original)
    b = measured(np.random.default_rng(2), 0.15, 0.08, original)
    r = judge(a, b, original, weight=10)
    flips = 100 * np.mean([(a.right[t] != original[t]).mean() for t in TASKS])
    assert r["score"] == pytest.approx(flips + 10 * a.kl)
    low, high = r["interval"]
    assert high - r["relative"] == pytest.approx(r["relative"] - low)
    assert (high - low) / 2 == pytest.approx(100 * 1.96 * r["standard_error"] / r["preset_score"])


def test_questions_both_files_flip_add_no_uncertainty(original) -> None:
    a = measured(np.random.default_rng(1), 0.1, 0.05, original)
    a.chunks = np.full(40, 0.05)
    flipped_together = Measured({t: ~original[t] for t in TASKS}, 0.05, 90.0, np.full(40, 0.05))
    assert judge(flipped_together, flipped_together, original)["standard_error"] == 0


def test_more_questions_narrow_the_interval() -> None:
    def width(n: int) -> float:
        rng = np.random.default_rng(0)
        original = {t: rng.random(n) < 0.6 for t in TASKS}
        r = judge(measured(rng, 0.1, 0.05, original), measured(rng, 0.12, 0.06, original), original)
        return r["interval"][1] - r["interval"][0]
    assert width(8000) < width(500) / 2


# --- GSM8K ---

def test_a_gsm8k_answer_is_right_when_the_number_after_the_hashes_matches() -> None:
    answer = "She has 3 + 4 = 7 apples.\n#### 7"
    assert correct("3 + 4 = 7\n#### 7", answer)
    assert correct("#### 7.", answer) and correct("####7", answer)
    assert not correct("#### 8", answer) and not correct("the answer is 7", answer)
    assert correct("#### 1,200", "#### 1200") and correct("#### $5", "#### 5")
    assert not correct("#### 7\n#### 8", "#### 8")  # the first #### counts


def write_gsm8k(folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"question": f"Tom has {i} apples and buys {i + 1} more. How many?",
             "answer": f"{i} + {i + 1}\n#### {2 * i + 1}"} for i in range(6)]
    questions, examples = folder / "gsm8k.jsonl", folder / "examples.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows[1:]) + "\n")
    examples.write_text(json.dumps(rows[0]) + "\n")
    return questions, examples


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Model:
    for module in (model_module, presets_module, verdict_module, winner_module):
        monkeypatch.setattr(module, "DATA", tmp_path)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    write_original(m.original)
    return m


def quantized(m: Model, preset: str) -> Path:
    out = m.dir / f"{preset}.gguf"
    subprocess.run(["llama-quantize", m.original, out, preset], check=True, capture_output=True)
    return out


@needs_perplexity
def test_llama_server_answers_every_prompt_and_stops(m: Model) -> None:
    resources = Resources()
    with LlamaServer(m.original, resources) as server:
        out = server.complete([f"Question: {i}?\nAnswer:" for i in range(10)])
        process = server.process
    assert len(out) == 10 and all(isinstance(o, str) for o in out)
    assert process.poll() is not None and resources.used_gpu == resources.used_ram == 0


@needs_perplexity
def test_gsm8k_gives_one_answer_per_question_and_records_the_originals_once(m: Model, tmp_path: Path) -> None:
    gsm8k = GSM8K(m, *write_gsm8k(tmp_path / "v"), Resources())
    assert len(gsm8k.answers(m.original)) == 5
    first = gsm8k.original_answers()
    before = gsm8k.record.stat().st_mtime_ns
    assert np.array_equal(gsm8k.original_answers(), first) and gsm8k.record.stat().st_mtime_ns == before


# --- the verdict, on the tiny model ---

@needs_perplexity
def test_the_verdict_on_the_tiny_model(m: Model, tmp_path: Path) -> None:
    text, questions = ga_set(tmp_path / "verdict")
    fitness = Fitness(m, text, questions)
    verdict = Verdict(fitness, GSM8K(m, *write_gsm8k(tmp_path / "verdict"), fitness.resources))
    preset = {"preset": "Q2_K", "file": str(quantized(m, "Q2_K")), "unfair": False}
    result = verdict.run(f"{m.name}@1", quantized(m, "Q8_0"), preset)
    assert result["verdict"] in ("WIN", "TIE", "LOSE") and result["unfair"] is False
    assert result["interval"][0] <= result["relative"] <= result["interval"][1]
    assert result["kl"]["ours"] < result["kl"]["preset"]  # Q8_0 drifts less than Q2_K
    assert set(result["accuracy"]["ours"]) == set(TASKS) and result["kl_weight_5"] in ("WIN", "TIE", "LOSE")
    again = verdict.run(f"{m.name}@1", Path("never measured"), preset)
    assert again == result  # once: read back, not measured again


# --- the winner ---

class StandIns:
    """Stand in for Genes (build writes the genome's name), Distill (copies, marking it distilled) and Fitness
    (a file's score from its contents)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def build(self, genome: dict, path: Path) -> Path:
        path.write_text(genome["name"])
        return path

    def run(self, file: Path, out: Path) -> list[float]:
        shutil.copy(file, out)
        out.write_text(file.read_text() + "-distilled")
        return []

    def score(self, file: Path) -> Score:
        s = self.scores[file.read_text()]
        return Score(s, s, 0.0, 100.0, {})


def test_the_lowest_of_every_finalist_built_and_distilled_wins_and_only_its_file_is_kept(m: Model) -> None:
    stand_ins = StandIns({"a": 5.0, "a-distilled": 3.0, "b": 4.0, "b-distilled": 6.0, "qat": 3.5})
    winner = Winner(f"{m.name}@1", stand_ins, stand_ins, stand_ins)
    extra = winner.dir / "qat.gguf"
    winner.dir.mkdir(parents=True)
    extra.write_text("qat")
    preset = {"preset": "Q2_K", "scores": {"Q2_K": {"score": 6.0, "flips": 6.0, "kl": 0.0, "same_top": 100.0,
                                                    "flips_per_task": {}}}}
    result = winner.run([{"genome": {"name": "a"}}, {"genome": {"name": "b"}}], preset, (extra,))
    assert result["was"] == "finalist-0-distilled.gguf" and result["score"]["score"] == 3.0
    assert result["relative"] == pytest.approx(50.0) and len(result["scores"]) == 5
    assert [f.name for f in winner.dir.glob("*.gguf")] == ["winner.gguf"]
    assert (winner.dir / "winner.gguf").read_text() == "a-distilled"
    assert winner.run([], preset) == result  # chosen once
