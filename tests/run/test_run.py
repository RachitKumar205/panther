"""The entrypoint: which modules it calls, in which order, with what; argument parsing. Every module is a recording
stand-in: no network, no GPU."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:  # the data module may not be finished in this checkout: stand in for it so panther.run imports
    from panther.data import Dataset  # noqa: F401
except ImportError:
    sys.modules["panther.data"] = types.SimpleNamespace(Dataset=None)

from panther import run as run_module  # noqa: E402


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple]:
    """Every module replaced by a stand-in that records (what, arguments)."""
    seen: list[tuple] = []
    sets = SimpleNamespace(
        distill=Path("distill.txt"), ga=SimpleNamespace(text=Path("ga.txt"), questions={"mmlu": Path("ga.bin")}),
        winner=SimpleNamespace(text=Path("winner.txt"), questions={"mmlu": Path("winner.bin")}),
        verdict=SimpleNamespace(text=Path("verdict.txt"), questions={"mmlu": Path("verdict.bin")}),
        gsm8k_questions=Path("gsm8k.jsonl"), gsm8k_examples=Path("examples.jsonl"),
        qat_tokens=np.zeros((8, 4)), qat_validation=np.zeros((2, 4)))

    class Dataset:
        web_distill = Path("web-distill.txt")

        def build(self):
            seen.append(("Dataset.build",))

        def for_model(self, m, qat_windows=1024):  # as many QAT windows as asked for
            seen.append(("Dataset.for_model", m.name))
            sets.qat_tokens = np.zeros((qat_windows, 4))
            return sets

    class Model:
        def __init__(self, name):
            self.name = name

        @classmethod
        def load(cls, size):
            seen.append(("Model.load", size))
            return cls("Qwen2.5-14B-Instruct")

        def download(self):
            seen.append(("Model.download",))

        def build(self, text):
            seen.append(("Model.build", text))

    def recorder(what, result=None):
        class Recorded:
            def __init__(self, *args):
                seen.append((what, *args))

            def __getattr__(self, method):
                def call(*args):
                    seen.append((f"{what}.{method}", *args))
                    return result(method, args) if result else None
                return call
        return Recorded

    def search_result(method, args):
        return [{"genome": {"a": "Q4_K"}}, {"genome": {"a": "Q3_K"}}, {"genome": {"a": "Q2_K"}}]

    class Search(recorder("Search", search_result)):
        genes = SimpleNamespace(build=lambda genome, path: (seen.append(("Genes.build", genome, path.name)),
                                                            path.write_text("x"))[1])

    def write_out(method, args):  # BlockAP.run(start, out), Distill.run(file, out): write out
        args[1].write_text("trained")

    monkeypatch.setattr(run_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_module, "DATA", tmp_path)
    monkeypatch.setattr(run_module, "Dataset", Dataset)
    monkeypatch.setattr(run_module, "Model", Model)
    monkeypatch.setattr(run_module, "Budget", recorder("Budget"))
    monkeypatch.setattr(run_module, "Fitness", recorder("Fitness"))
    monkeypatch.setattr(run_module, "Presets", recorder("Presets", lambda m, a: {
        "preset": "Q2_K", "file": "/presets/Q2_K.gguf", "unfair": False}))
    monkeypatch.setattr(run_module, "Search", Search)
    monkeypatch.setattr(run_module, "BlockAP", recorder("BlockAP", write_out))
    monkeypatch.setattr(run_module, "Distill", recorder("Distill", write_out))
    monkeypatch.setattr(run_module, "Winner", recorder("Winner", lambda m, a: {"file": "/contests/winner.gguf"}))
    monkeypatch.setattr(run_module, "GSM8K", recorder("GSM8K"))
    monkeypatch.setattr(run_module, "Verdict", recorder("Verdict", lambda m, a: {
        "verdict": "WIN", "relative": 91.2, "interval": [86.0, 96.4], "unfair": False}))
    return seen


def names(calls: list[tuple]) -> list[str]:
    return [c[0] for c in calls]


def test_fetch_builds_the_dataset_and_downloads_the_model(calls) -> None:
    run_module.main(["fetch", "14b"])
    assert calls == [("Dataset.build",), ("Model.load", "14b"), ("Model.download",)]


def test_run_calls_the_parts_in_order_with_the_right_sets(calls, capsys) -> None:
    run_module.main(["run", "14b@5"])
    assert names(calls) == ["Model.load", "Model.build", "Dataset.for_model", "Budget",
                            "Presets", "Fitness", "Presets.best",
                            "Fitness", "Search", "Search.run",
                            "Distill", "Fitness", "Winner", "Winner.run",
                            "Fitness", "GSM8K", "Verdict", "Verdict.run"]
    fitness = [c[2] for c in calls if c[0] == "Fitness"]  # the text each Fitness was made with
    assert fitness == [Path("winner.txt"), Path("ga.txt"), Path("winner.txt"), Path("verdict.txt")]
    assert ("Model.build", Path("web-distill.txt")) in calls  # GPTQ and the importance matrix: the distill set
    winner_run = next(c for c in calls if c[0] == "Winner.run")
    assert winner_run[3] == ()  # no QAT file
    assert next(c for c in calls if c[0] == "Winner")[1] == "Qwen2.5-14B-Instruct@5"
    verdict_run = next(c for c in calls if c[0] == "Verdict.run")
    assert verdict_run[1:3] == ("Qwen2.5-14B-Instruct@5", Path("/contests/winner.gguf"))
    assert "WIN: relative 91.2 (95 % interval 86.0–96.4" in capsys.readouterr().out


def test_qat_trains_the_best_finalist_distills_it_and_adds_it_to_the_winners_files(calls, tmp_path) -> None:
    run_module.main(["run", "14b@5", "--qat", "4"])
    order = names(calls)
    assert order.index("Search.run") < order.index("BlockAP") < order.index("Winner.run")
    build = next(c for c in calls if c[0] == "Genes.build")
    assert build[1] == {"a": "Q4_K"}  # the best finalist
    blockap = next(c for c in calls if c[0] == "BlockAP")
    assert blockap[2].shape == (4, 4)  # its first N samples
    folder = tmp_path / "contests" / "Qwen2.5-14B-Instruct@5"
    assert next(c for c in calls if c[0] == "BlockAP.run")[1:] == (folder / "finalist-0.gguf", folder / "qat.gguf")
    assert next(c for c in calls if c[0] == "Winner.run")[3] == (folder / "qat-distilled.gguf",)
    assert not (folder / "qat.gguf").exists()  # the undistilled QAT file is only a step


def test_qat_done_before_is_not_redone(calls, tmp_path) -> None:
    folder = tmp_path / "contests" / "Qwen2.5-14B-Instruct@5"
    folder.mkdir(parents=True)
    (folder / "qat-distilled.gguf").write_text("done")
    run_module.main(["run", "14b@5", "--qat", "4"])
    assert "BlockAP" not in names(calls)


@pytest.mark.parametrize("bad", ["14b", "14@5", "b@5", "14b@", "14b@5.5", "big@5"])
def test_a_badly_written_contest_fails(bad: str) -> None:
    with pytest.raises(SystemExit):
        run_module.main(["run", bad])


def test_a_contest_is_read_as_size_and_budget() -> None:
    assert run_module.contest("14B@5") == ("14b", 5) and run_module.contest("1.5b@1") == ("1.5b", 1)


def test_a_run_without_a_gpu_fails_before_doing_anything(calls, monkeypatch) -> None:
    monkeypatch.setattr(run_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError, match="no GPU"):
        run_module.main(["run", "14b@5"])
    assert calls == []


def test_qat_is_not_trained_again_once_the_winner_is_chosen(calls, tmp_path) -> None:
    folder = tmp_path / "contests" / "Qwen2.5-14B-Instruct@5"
    folder.mkdir(parents=True)
    (folder / "winner.json").write_text("{}")  # chosen; the losing QAT file was deleted
    run_module.main(["run", "14b@5", "--qat", "4"])
    assert "BlockAP" not in names(calls)
