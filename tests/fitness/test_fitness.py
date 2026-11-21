"""Fitness: the score's arithmetic, parsing llama-perplexity's output, and real runs on the tiny model."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from panther.fitness import KL_WEIGHT, Fitness, Score
from panther.fitness.resources import Resources
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
    monkeypatch.setattr(fitness, "perplexity", lambda *a: "noise\n" + "\n".join(lines) + "\nmore noise")
    assert fitness.answers(Path("x.gguf"), "mmlu").tolist() == right


def test_winogrande_lines_give_chosen_against_correct(fitness: Fitness, monkeypatch) -> None:
    lines = [f"{i}\t50.0000\t -5.6   -5.7  {c}  {r}" for i, (c, r) in enumerate([(1, 1), (1, 2)] * 5, 1)]
    monkeypatch.setattr(fitness, "perplexity", lambda *a: "\n".join(lines))
    assert fitness.answers(Path("x.gguf"), "winogrande").tolist() == [True, False] * 5


def test_answers_for_fewer_questions_than_the_file_holds_fail(fitness: Fitness, monkeypatch) -> None:
    monkeypatch.setattr(fitness, "perplexity", lambda *a: "1\t100.00000000\n2\t50.00000000")
    with pytest.raises(AssertionError, match="2 of 12 questions"):
        fitness.answers(Path("x.gguf"), "mmlu")


def test_each_chunks_kl_is_recovered_from_the_running_mean_and_every_chunk_is_read(fitness, monkeypatch) -> None:
    monkeypatch.setattr(fitness, "kl_base", lambda: Path("base.kld"))
    out = (Path(__file__).parents[1] / "fixtures/kl_q6k_48chunks.log").read_text()
    monkeypatch.setattr(fitness, "perplexity", lambda *a: out)
    mean, _, each = fitness.kl_chunks(Path("x.gguf"))
    assert len(each) == 48 and each[0] == pytest.approx(0.00634)
    assert each.mean() == pytest.approx(mean, abs=1e-5)  # the chunks average to the summary's mean
    assert each.min() > 0.003 and each.max() < 0.013  # each chunk's own KL, not the running mean (~0.0058)
    negative = out.replace("       0.00740 ±", "      -0.00740 ±", 1)  # a chunk where the file beats the original
    monkeypatch.setattr(fitness, "perplexity", lambda *a: negative)
    assert len(fitness.kl_chunks(Path("x.gguf"))[2]) == 48
    nan = out.replace("0.00740 ±    0.00355", "0.00740 ±       -nan", 1)  # llama.cpp prints -nan for one chunk's ±
    monkeypatch.setattr(fitness, "perplexity", lambda *a: nan)
    assert len(fitness.kl_chunks(Path("x.gguf"))[2]) == 48
    monkeypatch.setattr(fitness, "perplexity", lambda *a: out.replace("  48 ", "  # ", 1))
    with pytest.raises(AssertionError, match="47 of 48 chunks"):
        fitness.kl_chunks(Path("x.gguf"))


def test_kl_and_same_top_are_read_from_llama_perplexitys_summary(fitness: Fitness, monkeypatch) -> None:
    monkeypatch.setattr(fitness, "kl_base", lambda: Path("base.kld"))
    out = (Path(__file__).parents[1] / "fixtures/kl_q6k_48chunks.log").read_text()
    monkeypatch.setattr(fitness, "perplexity", lambda *a: out)
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


# --- running at once ---

def test_a_files_measurements_run_at_once(fitness: Fitness, monkeypatch) -> None:
    running, most = [0], [0]
    lock = threading.Lock()
    right = "\n".join(f"{i}\t100.00000000" for i in range(1, 13))
    wino = "\n".join(f"{i}\t50.0\t -5.6   -5.7  1  1" for i in range(1, 11))

    def measure(file, positions, *args):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return "Mean    KLD:   0.1 ± 0.0\nSame top p: 90.0 ± 0.1 %" if "--kl-divergence" in args else \
            wino if "--winogrande" in args else right
    monkeypatch.setattr(fitness, "perplexity", measure)
    monkeypatch.setattr(fitness, "kl_base", lambda: Path("base.kld"))
    monkeypatch.setattr(fitness, "original_answers", lambda task: np.ones(10 if task == "winogrande" else 12, bool))
    assert fitness.score(Path("x.gguf")).kl == 0.1
    assert most[0] == 5  # KL and four tasks together


def test_jobs_wait_for_their_share_and_one_larger_than_everything_runs_alone() -> None:
    r = Resources()
    r.gpu, r.ram = 10, 10
    log = []

    def job(name: str, gpu: int, ram: int) -> None:
        with r.hold(gpu, ram):
            log.append(f"{name} in")
            time.sleep(0.05)
            log.append(f"{name} out")
    threads = [threading.Thread(target=job, args=a) for a in (("a", 6, 1), ("b", 6, 1), ("c", 20, 20))]
    for t in threads:
        t.start()
        time.sleep(0.01)
    for t in threads:
        t.join()
    assert log.index("a out") < log.index("b in")  # b's GPU share was only free once a finished
    c_in = log.index("c in")  # c is larger than everything: it ran alone (b may go before or after it)
    assert log[c_in + 1] == "c out" and all(log.index(f"{x} out") < c_in for x in "ab" if log.index(f"{x} in") < c_in)
    assert r.used_gpu == r.used_ram == 0
