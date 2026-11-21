"""The Model: its config and files, the commands each step runs, and the type copies end to end with llama.cpp."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from panther.model import model as model_module
from panther.model.gptq import GPTQ, input_of
from panther.model.model import NEEDS_IMPORTANCE, TYPES, Model, bits
from panther.model.quantizer import Quantizer
from tests.model.tiny import fake_hessians, needs_llama, write_imatrix, write_original

NAMES = {"Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-22B-Instruct",
         "Mistral-24B-Instruct"}


# --- config and files ---

def test_thirteen_types_fewest_bits_first_no_two_the_same_size() -> None:
    sizes = [bits(t) for t in TYPES]
    assert len(TYPES) == 13 and "IQ3_S" not in TYPES
    assert sizes == sorted(sizes) and len(set(sizes)) == 13


def test_every_model_loads_by_name_and_size_pinned_to_a_commit() -> None:
    for name in NAMES:
        m = Model.load(name)
        assert Model.load(name.split("-")[1].lower()) == m
        assert len(m.revision) == 40


def test_an_unknown_model_fails() -> None:
    with pytest.raises(ValueError):
        Model.load("3b")


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Model:
    monkeypatch.setattr(model_module, "DATA", tmp_path)
    monkeypatch.setattr(Model, "require_disk", lambda self, need: None)
    return Model.load("1.5b")


def test_files_live_in_the_models_folder(m: Model, tmp_path: Path) -> None:
    assert m.original == tmp_path / "models/Qwen2.5-1.5B-Instruct/original.gguf"
    assert m.copy("Q4_K") == tmp_path / "models/Qwen2.5-1.5B-Instruct/types/Q4_K.gguf"


def test_too_little_disk_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "DATA", tmp_path)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="free disk"):
        m.require_disk(10**18)


# --- the steps' commands (recorded, not run) ---

@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Each program is recorded and 'writes' its output file."""
    seen = []

    def fake(cmd, check):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        Path(cmd[cmd.index("--outfile" if "--outfile" in cmd else "-o") + 1]).touch()
    monkeypatch.setattr(subprocess, "run", fake)
    return seen


def test_convert_and_imatrix_write_then_rename_and_are_skipped_when_done(m: Model, commands, tmp_path: Path) -> None:
    text = tmp_path / "train.txt"
    m.hf.mkdir(parents=True)  # downloaded
    m.convert()
    m.make_imatrix(text)
    assert m.original.exists() and m.imatrix.exists() and not list(m.dir.glob("*.part*"))
    assert commands[0][commands[0].index("--outtype") + 1] == "bf16"
    assert commands[1][:2] == ["llama-imatrix", "-m"] and commands[1][commands[1].index("-f") + 1] == str(text)
    assert "--process-output" in commands[1]  # the output layer gets importance data too
    m.convert()
    m.make_imatrix(text)
    assert len(commands) == 2


def test_download_is_pinned_and_happens_once(m: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def download(repo, **kw):
        calls.append((repo, kw))
        kw["local_dir"].mkdir(parents=True)
    monkeypatch.setattr(model_module, "snapshot_download", download)
    monkeypatch.setattr(model_module, "HfApi", lambda: SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(safetensors=SimpleNamespace(total=1_543_714_304))))
    m.download()
    m.download()
    assert calls == [(m.repo, {"revision": m.revision, "local_dir": m.hf,
                               "ignore_patterns": ["consolidated*", "original/*", "*.pth"]})]


def test_build_runs_the_steps_in_order_and_keeps_the_download(m: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    order = []
    monkeypatch.setattr(Model, "download", lambda self: order.append("download"))
    monkeypatch.setattr(Model, "convert", lambda self: order.append("convert"))
    monkeypatch.setattr(Model, "make_imatrix", lambda self, text: order.append("imatrix"))
    monkeypatch.setattr(Model, "make_type_copies", lambda self, text: order.append("copies"))
    m.hf.mkdir(parents=True)
    m.build(Path("train.txt"))
    assert order == ["download", "convert", "imatrix", "copies"] and m.hf.exists()


# --- type copies, end to end: a tiny Qwen2-shaped model, real llama-quantize and quantizer ---

@pytest.fixture
def stored(m: Model, monkeypatch: pytest.MonkeyPatch) -> Model:
    m.dir.mkdir(parents=True)
    write_original(m.original)
    write_imatrix(m.original, m.imatrix)
    monkeypatch.setattr(Model, "hessians", fake_hessians)
    return m


@needs_llama
def test_every_type_copy_holds_gptq_codes_and_the_original_elsewhere(stored: Model) -> None:
    stored.make_type_copies(Path("unused"))
    original = {x.name: x for x in gguf.GGUFReader(stored.original).tensors}
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the device the build used
    assert not list(stored.types.glob("*.part*"))
    for t in TYPES:
        copy = {x.name: x for x in gguf.GGUFReader(stored.copy(t)).tensors}
        assert copy.keys() == original.keys()
        assert copy["token_embd.weight"].tensor_type.name == ("Q8_0" if t in NEEDS_IMPORTANCE else t)
        for name, x in original.items():
            if len(x.shape) == 2 and name != "token_embd.weight":  # every layer matrix and the output layer
                assert copy[name].tensor_type.name == t
            elif len(x.shape) == 1:  # norms and biases: as they were
                assert np.array_equal(copy[name].data, x.data)
        for name in ("blk.1.ffn_down.weight", "output.weight"):  # in full: their bytes are GPTQ's codes
            [h] = next(fake_hessians(stored, None, [input_of(name)])).values()
            w = torch.from_numpy(gguf.quants.dequantize(original[name].data, original[name].tensor_type))
            assert np.array_equal(copy[name].data, GPTQ(h.to(device)).round(w.to(device), t, Quantizer()))


@needs_llama
def test_a_tied_model_has_no_output_layer_to_round(stored: Model) -> None:
    write_original(stored.original, tied=True)
    write_imatrix(stored.original, stored.imatrix)
    stored.make_type_copies(Path("unused"))
    assert "output.weight" not in {x.name for x in gguf.GGUFReader(stored.copy("Q4_K")).tensors}


@needs_llama
def test_a_stopped_build_leaves_no_copy_and_is_redone(stored: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    def stops(self, text, modules):
        yield next(fake_hessians(self, text, modules))
        raise KeyboardInterrupt
    monkeypatch.setattr(Model, "hessians", stops)
    with pytest.raises(KeyboardInterrupt):
        stored.make_type_copies(Path("unused"))
    assert not any(stored.copy(t).exists() for t in TYPES)
    monkeypatch.setattr(Model, "hessians", fake_hessians)
    stored.make_type_copies(Path("unused"))
    assert all(stored.copy(t).exists() for t in TYPES)
    assert not list(stored.copy("Q8_0").parent.glob("*.part*"))


@needs_llama
def test_a_matrix_gptq_never_reached_fails_the_build(stored: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Model, "hessians", lambda self, text, modules: [next(fake_hessians(self, text, modules))])
    with pytest.raises(AssertionError, match="no GPTQ codes"):
        stored.make_type_copies(Path("unused"))


@needs_llama
def test_existing_copies_are_kept(stored: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    stored.make_type_copies(Path("unused"))
    before = stored.copy("Q4_K").stat().st_mtime_ns
    monkeypatch.setattr(Model, "hessians", lambda self, text, modules: pytest.fail("rebuilt"))
    stored.make_type_copies(Path("unused"))
    assert stored.copy("Q4_K").stat().st_mtime_ns == before
