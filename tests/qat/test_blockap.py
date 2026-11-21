"""Block-AP on a tiny model (a transformers model and the same weights as its 16-bit GGUF; the start file made by
the real llama-quantize): never worse than the file's block, only layer matrices change, llama.cpp runs the result,
a stopped run continues, Mistral's q/k row order."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from panther.distill.distill import permute, unpermute
from panther.fitness import Fitness
from panther.model import Model
from panther.model import model as model_module
from panther.qat import BlockAP, Grid
from panther.qat import blockap as blockap_module
from panther.qat.blockap import QLinear
from tests.distill.tiny_hf import make
from tests.fitness.questions import write_text
from tests.model.tiny import HEADS, needs_llama, needs_perplexity, write_imatrix

pytestmark = needs_llama
LENGTH = 256  # tokens per sample (2,048 on a real model)


def llama_tokens(m: Model, text: Path) -> np.ndarray:
    out = subprocess.run(["llama-tokenize", "-m", m.original, "-f", text, "--ids", "--log-disable"], check=True,
                         capture_output=True, text=True).stdout
    return np.array(json.loads(out.strip().splitlines()[-1]))


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llama: bool = False, preset: str = "Q2_K",
          disk: bool = False):
    """The tiny model, a start file in `preset`, 16 training and 4 validation samples of real text."""
    monkeypatch.setattr(model_module, "DATA", tmp_path / "data")
    if not disk:  # the tiny model needs a few MB, not the 20 GiB margin
        monkeypatch.setattr(Model, "require_disk", lambda self, need: None)
    monkeypatch.setattr(blockap_module, "CHECK", 4)  # checks every 4 steps (512 on a real model)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    make(m.hf, m.original, llama=llama)
    write_imatrix(m.original, m.imatrix)
    start = m.dir / f"{preset}.gguf"
    subprocess.run(["llama-quantize", "--pure", "--imatrix", m.imatrix, "--token-embedding-type", "q8_0", m.original,
                    start, preset], check=True, capture_output=True)
    ids = llama_tokens(m, write_text(tmp_path / "qat.txt", words=12000, seed=5))
    samples = ids[: 20 * LENGTH].reshape(20, LENGTH)
    return m, start, BlockAP(m, samples[:16], samples[16:])


def test_a_matrix_trains_in_the_files_row_order_and_runs_in_transformers_order() -> None:
    rows, cols = 64, 256
    linear = torch.nn.Linear(cols, rows, bias=False)
    file_values = permute(linear.weight.detach(), 4)  # how llama.cpp stores a q matrix of 4 heads
    step = 1 / 64  # exact in 16 bits
    q8 = torch.clamp((file_values / step).round(), -128, 127).long().view(rows, cols // 32, 32)
    codes = Grid("Q8_0").pack({"d": torch.full((rows, cols // 32), step), "q": q8})
    q = QLinear(linear, Grid("Q8_0"), codes, heads=4)
    assert torch.equal(q.W.detach(), file_values)  # trained in the file's order
    x = torch.randn(3, cols)
    decoded = torch.from_numpy(gguf.quants.dequantize(codes, gguf.GGMLQuantizationType.Q8_0))
    assert torch.allclose(q(x), torch.nn.functional.linear(x, unpermute(decoded, 4)), atol=1e-5)  # run in HF order
    assert HEADS == 4


def test_too_little_disk_fails_before_anything_starts(tmp_path, monkeypatch) -> None:
    m, start, qat = setup(tmp_path, monkeypatch, disk=True)
    monkeypatch.setattr(model_module.shutil, "disk_usage", lambda p: SimpleNamespace(free=10))
    with pytest.raises(RuntimeError, match="free disk"):
        qat.run(start, tmp_path / "out.gguf")


@needs_perplexity
def test_every_block_ends_no_worse_than_the_files_and_only_layer_matrices_change(tmp_path, monkeypatch) -> None:
    m, start, qat = setup(tmp_path, monkeypatch)
    out = tmp_path / "out.gguf"
    log = qat.run(start, out)
    assert len(log) == 2 and all(b["after"] <= b["file"] for b in log)
    assert any(not b["kept_file"] and b["after"] < b["file"] for b in log)  # training did beat the file somewhere
    assert out.stat().st_size == start.stat().st_size
    before = {t.name: t for t in gguf.GGUFReader(start).tensors}
    after = {t.name: t for t in gguf.GGUFReader(out).tensors}
    for name, t in before.items():
        if not (name.startswith("blk.") and len(t.shape) == 2):
            assert np.array_equal(t.data, after[name].data), name  # word table, output layer, norms, biases
        assert t.tensor_type == after[name].tensor_type
    fitness = Fitness(m, write_text(tmp_path / "kl.txt", seed=9), {})
    print(f"\nKL: start {fitness.kl(start)[0]:.5f}, after Block-AP {fitness.kl(out)[0]:.5f}")
    assert fitness.kl(out)[0] <= fitness.kl(start)[0] * 1.05  # llama.cpp runs it, and it is not worse


def test_a_stopped_run_continues_after_its_last_finished_block(tmp_path, monkeypatch) -> None:
    m, start, qat = setup(tmp_path, monkeypatch)
    train = BlockAP.train

    def stops_at_block_1(self, net, b):
        if b == 1:
            raise KeyboardInterrupt
        return train(self, net, b)
    monkeypatch.setattr(BlockAP, "train", stops_at_block_1)
    with pytest.raises(KeyboardInterrupt):
        qat.run(start, tmp_path / "out.gguf")
    saved = np.load(qat.block_file(0))
    first = {k: saved[k].copy() for k in saved}
    trained = []
    monkeypatch.setattr(BlockAP, "train", lambda self, net, b: trained.append(b) or train(self, net, b))
    log = qat.run(start, tmp_path / "out.gguf")
    assert trained == [1] and log[0] == {"block": 0, "resumed": True}  # block 0 was not trained again
    after = {t.name: t for t in gguf.GGUFReader(tmp_path / "out.gguf").tensors}
    assert all(np.array_equal(after[k].data.reshape(-1), v.reshape(-1)) for k, v in first.items())


def test_another_start_file_gets_its_own_work_folder(tmp_path, monkeypatch) -> None:
    m, start, qat = setup(tmp_path, monkeypatch)
    qat.run(start, tmp_path / "a.gguf")
    other = m.dir / "other.gguf"
    subprocess.run(["llama-quantize", "--pure", m.original, other, "Q4_K_S"], check=True, capture_output=True)
    trained = []
    train = BlockAP.train
    monkeypatch.setattr(BlockAP, "train", lambda self, net, b: trained.append(b) or train(self, net, b))
    qat.run(other, tmp_path / "b.gguf")
    assert trained == [0, 1]  # nothing reused from the first file's run
    assert len(list((m.dir / "qat").iterdir())) == 2


def test_a_llama_style_model_trains_with_its_q_and_k_rows_in_order(tmp_path, monkeypatch) -> None:
    m, start, qat = setup(tmp_path, monkeypatch, llama=True, preset="Q4_K_S")
    log = qat.run(start, tmp_path / "out.gguf")
    # with the rows in the wrong order the block would be far worse than the file's and always kept as it was
    assert any(not b["kept_file"] and b["after"] < b["file"] for b in log)
