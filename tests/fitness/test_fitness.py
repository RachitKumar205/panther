"""Fitness: the score's arithmetic, parsing llama-perplexity's output, and real runs on the tiny model."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from panther.fitness import KL_WEIGHT, Fitness, Score
from panther.fitness import fitness as fitness_module
from panther.model import Model
from panther.model import model as model_module
from tests.fitness.questions import ga_set, write_multiple_choice, write_text, write_winogrande
from tests.model.tiny import needs_perplexity, write_original


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Model:
    monkeypatch.setattr(model_module, "DATA", tmp_path)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    write_original(m.original)
    return m


@pytest.fixture
def fitness(m: Model, tmp_path: Path) -> Fitness:
    return Fitness(m, *ga_set(tmp_path / "ga"))


def quantized(m: Model, preset: str) -> Path:
    out = m.dir / f"{preset}.gguf"
    subprocess.run(["llama-quantize", m.original, out, preset], check=True, capture_output=True)
    return out


# --- the arithmetic and the parsing (no llama.cpp) ---

def test_relative_score_is_a_share_of_the_preset() -> None:
    ours, preset = Score(5.0, 4.0, 0.1, 90.0, {}), Score(10.0, 8.0, 0.2, 80.0, {})
    assert ours.relative(preset) == 50.0 and preset.relative(preset) == 100.0


def test_running_accuracy_gives_each_questions_right_or_wrong(fitness: Fitness, monkeypatch, tmp_path: Path) -> None:
    right = [True, False, False, True, True, False, True, True, False, False, False, True]
    lines = [f"{i}\t{100 * sum(right[:i]) / i:.8f}" for i in range(1, 13)]
    monkeypatch.setattr(fitness_module, "perplexity", lambda *a: "noise\n" + "\n".join(lines) + "\nmore noise")
    assert fitness.answers(Path("x.gguf"), "mmlu").tolist() == right


def test_winogrande_lines_give_chosen_against_correct(fitness: Fitness, monkeypatch) -> None:
    lines = [f"{i}\t50.0000\t -5.6   -5.7  {c}  {r}" for i, (c, r) in enumerate([(1, 1), (1, 2)] * 5, 1)]
    monkeypatch.setattr(fitness_module, "perplexity", lambda *a: "\n".join(lines))
    assert fitness.answers(Path("x.gguf"), "winogrande").tolist() == [True, False] * 5


def test_answers_for_fewer_questions_than_the_file_holds_fail(fitness: Fitness, monkeypatch) -> None:
    monkeypatch.setattr(fitness_module, "perplexity", lambda *a: "1\t100.00000000\n2\t50.00000000")
    with pytest.raises(AssertionError, match="2 of 12 questions"):
        fitness.answers(Path("x.gguf"), "mmlu")


def test_kl_and_same_top_are_read_from_llama_perplexitys_summary(fitness: Fitness, monkeypatch) -> None:
    monkeypatch.setattr(fitness, "kl_base", lambda: Path("base.kld"))
    out = (Path(__file__).parents[1] / "fixtures/kl_q6k_48chunks.log").read_text()
    monkeypatch.setattr(fitness_module, "perplexity", lambda *a: out)
    assert fitness.kl(Path("x.gguf")) == (0.005810, 95.453)


# --- real llama-perplexity on the tiny model ---

@needs_perplexity
def test_the_original_scores_zero(fitness: Fitness, m: Model) -> None:
    s = fitness.score(m.original)
    assert s.flips == 0 and s.kl == pytest.approx(0, abs=1e-5) and s.score == pytest.approx(0, abs=1e-4)
    assert set(s.flips_per_task) == {"mmlu", "hellaswag", "arc_challenge", "winogrande"}


@needs_perplexity
def test_a_quantized_file_drifts_and_its_score_adds_up(fitness: Fitness, m: Model) -> None:
    s = fitness.score(quantized(m, "Q2_K"))
    assert s.kl > 0 and s.same_top < 100
    assert s.flips == pytest.approx(np.mean(list(s.flips_per_task.values())))
    assert s.score == pytest.approx(s.flips + KL_WEIGHT * s.kl)
    assert s.kl > fitness.score(quantized(m, "Q8_0")).kl  # fewer bits, more drift


@needs_perplexity
def test_every_question_is_answered_in_file_order(fitness: Fitness, m: Model) -> None:
    assert len(fitness.answers(m.original, "mmlu")) == 12
    assert len(fitness.answers(m.original, "winogrande")) == 10


@needs_perplexity
def test_the_originals_answers_and_probabilities_are_recorded_once(fitness: Fitness, m: Model) -> None:
    fitness.score(m.original)
    recorded = {f: f.stat().st_mtime_ns for f in fitness.reference.iterdir()}
    assert len(recorded) == 5  # one KL base, four tasks' answers
    fitness.score(quantized(m, "Q4_K_M"))
    assert {f: f.stat().st_mtime_ns for f in fitness.reference.iterdir()} == recorded


@needs_perplexity
def test_another_text_or_question_file_gets_its_own_record(fitness: Fitness, m: Model, tmp_path: Path) -> None:
    other = Fitness(m, write_text(tmp_path / "other.txt", seed=1),
                    {"mmlu": write_multiple_choice(tmp_path / "other.bin", 12, seed=9),
                     "winogrande": write_winogrande(tmp_path / "other.csv", 10, seed=9)})
    fitness.score(m.original)
    other.score(m.original)
    assert len(list(fitness.reference.iterdir())) == 5 + 3
