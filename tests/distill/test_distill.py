"""Distillation on a tiny model: loading the student exactly, the teacher's record, writing back bit for bit, and
that training lowers the drift llama.cpp measures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from torch.nn.utils import parametrize

from panther.distill import Distill
from panther.distill import distill as distill_module
from panther.distill.distill import BLOCK, SCALE_AT, kl_top, permute, unpermute
from panther.fitness import Fitness
from panther.model import Model
from panther.model import model as model_module
from tests.distill.tiny_hf import ROLES, llama_cpp_permute, make
from tests.fitness.questions import write_text
from tests.model.tiny import HEADS, KV_HEADS, needs_llama, needs_perplexity, write_imatrix


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Model:
    """The tiny Qwen2 model: its download (hf/), its 16-bit original and importance matrix; 16 sequences of text."""
    monkeypatch.setattr(model_module, "DATA", tmp_path)
    monkeypatch.setattr(distill_module, "HEADROOM", 0)  # the tiny model fits in 32 bits even on a small GPU
    monkeypatch.setattr(Model, "tokens", lambda self, text: np.random.default_rng(0).integers(0, 200, 16 * 512))
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    make(m.hf, m.original)
    write_imatrix(m.original, m.imatrix)
    return m


@pytest.fixture
def distill(m: Model, tmp_path: Path) -> Distill:
    return Distill(m, write_text(tmp_path / "distill.txt"))


def quantized(m: Model, preset: str) -> Path:
    out = m.dir / f"{preset}.gguf"
    subprocess.run(["llama-quantize", "--pure", "--imatrix", m.imatrix, "--token-embedding-type", "q8_0", m.original,
                    out, preset], check=True, capture_output=True)  # the word table has no importance data
    return out


def decoded(file: Path) -> dict[str, np.ndarray]:
    return {t.name: gguf.quants.dequantize(t.data, t.tensor_type) for t in gguf.GGUFReader(file).tensors}


# --- llama.cpp's row order ---

def test_permute_is_llama_cpps_and_unpermute_undoes_it() -> None:
    rows = np.arange(64 * 3).reshape(64, 3)
    assert np.array_equal(permute(rows, 4), llama_cpp_permute(rows, 4, 4))
    assert np.array_equal(permute(rows, 2), llama_cpp_permute(rows, 4, 2))  # k: by the KV heads
    assert np.array_equal(unpermute(permute(rows, 4), 4), rows)


# --- the student ---

@needs_llama
def test_the_student_holds_exactly_the_files_values(distill: Distill, m: Model) -> None:
    file = quantized(m, "Q4_K_S")
    net = distill.student(file)
    values = decoded(file)
    for name, module in distill.modules(net).items():
        weight = module.parametrizations.weight.original if parametrize.is_parametrized(module) else module.weight
        assert weight.dtype == torch.float32  # the tiny model fits on the GPU in 32 bits: exact
        assert np.array_equal(weight.detach().cpu().numpy(), values[name].reshape(weight.shape)), name


def test_a_llama_style_files_q_and_k_rows_are_put_back_in_order(m: Model, tmp_path: Path) -> None:
    net = make(m.hf, m.original, llama=True)  # a Mistral-like model: llama.cpp reorders its q and k rows
    distill = Distill(m, write_text(tmp_path / "t.txt"))
    student = distill.modules(distill.student(m.original))
    for name, module in distill.modules(net).items():
        assert torch.equal(module.weight, student[name].weight.detach().cpu()), name


@needs_llama
def test_only_block_scales_and_norm_shifts_train_and_not_the_word_table(distill: Distill, m: Model) -> None:
    net = distill.student(quantized(m, "Q4_K_S"))
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert all(n.endswith(("theta", "shift")) for n in trainable)
    assert not parametrize.is_parametrized(net.model.embed_tokens)
    assert parametrize.is_parametrized(net.lm_head)  # the output layer is trained too
    theta = net.model.layers[0].mlp.down_proj.parametrizations.weight[0].theta
    assert theta.shape == (256, 512 // BLOCK) and theta.dtype == torch.float32
    assert len(trainable) == 2 * 7 + 1 + 2 * 2 + 1  # 14 layer matrices, the output layer, 4 layer norms, the last


@needs_llama
def test_a_type_without_a_single_scale_is_not_trained(distill: Distill, m: Model) -> None:
    net = distill.student(quantized(m, "IQ1_M"))
    assert not any(n.endswith("theta") for n, _ in net.named_parameters())


# --- the teacher ---

def test_the_teacher_records_its_top_64_once(distill: Distill) -> None:
    t = distill.teacher()
    assert t["ids"].shape == (16, 512) and t["top_ids"].shape == t["top_logp"].shape == (16, 512, 64)
    assert t["top_logp"].dtype == torch.float32
    assert torch.all(t["top_logp"][..., :-1] >= t["top_logp"][..., 1:])  # most likely first
    before = distill.record.stat().st_mtime_ns
    assert torch.equal(distill.teacher()["top_ids"], t["top_ids"])
    assert distill.record.stat().st_mtime_ns == before


# --- writing back ---

@needs_llama
@pytest.mark.parametrize("preset", ["Q2_K", "Q3_K_S", "Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0", "IQ2_XS", "IQ4_XS"])
def test_writing_back_changes_only_the_scales_and_norms_and_exactly_as_trained(distill, m, tmp_path, preset) -> None:
    file = quantized(m, preset)
    net = distill.student(file)
    rng = torch.Generator().manual_seed(0)
    for p in net.parameters():
        if p.requires_grad:
            p.data = (torch.rand(p.shape, generator=rng) * 0.2 - 0.1).to(p.device)
    out = distill.write(file, tmp_path / "out.gguf", net)
    assert out.stat().st_size == file.stat().st_size
    before = {t.name: t for t in gguf.GGUFReader(file).tensors}
    after = {t.name: t for t in gguf.GGUFReader(out).tensors}
    for name, module in distill.modules(net).items():
        if name not in before:
            continue
        old, new = before[name], after[name]
        if not parametrize.is_parametrized(module):
            assert np.array_equal(old.data, new.data), name  # untouched, byte for byte
        elif name.endswith("norm.weight"):  # the file's 32-bit norm + the shift
            shift = module.parametrizations.weight[0].shift.detach().cpu().numpy()
            assert np.array_equal(new.data, old.data + shift), name
        else:
            weights, nbytes = gguf.GGML_QUANT_SIZES[old.tensor_type]
            a = np.asarray(old.data).reshape(old.data.shape[0], -1, nbytes)
            b = np.asarray(new.data).reshape(a.shape)
            scale = np.zeros(nbytes, bool)
            for at in SCALE_AT[old.tensor_type.name]:
                scale[at:at + 2] = True
            assert np.array_equal(a[:, :, ~scale], b[:, :, ~scale]), name  # the codes never change
            factor = torch.exp(module.parametrizations.weight[0].theta).detach().cpu().numpy()  # as the code does
            factor = np.repeat(factor, BLOCK // weights, axis=1)
            for at in SCALE_AT[old.tensor_type.name]:  # each scale: d × exp(θ), rounded to its 16 bits
                d = a[:, :, at:at + 2].copy().view(np.float16)[..., 0]
                want = (d.astype(np.float32) * factor).astype(np.float16)
                assert np.array_equal(b[:, :, at:at + 2].copy().view(np.float16)[..., 0], want), name
            trained = module.weight.detach().float().cpu().numpy()  # the student's weights × exp(θ)
            written = gguf.quants.dequantize(new.data, new.tensor_type)
            # equal up to the 16-bit scales. A scale this small (Q6_K, IQ4_XS: d ≈ 1e-5 for weights of 0.05) sits
            # below fp16's normal range, where it has fewer bits: measured up to 0.27 % of a row's largest weight.
            # llama.cpp's own quantizer writes d with the same limit.
            assert np.all(np.abs(written - trained) <= 2**-8 * np.abs(trained).max(axis=1, keepdims=True)), name


def test_a_llama_style_files_scales_are_written_in_its_own_row_order(m: Model, tmp_path: Path) -> None:
    make(m.hf, m.original, llama=True)
    file = tmp_path / "q8.gguf"  # the original with every matrix in Q8_0, by gguf-py
    reader = gguf.GGUFReader(m.original)
    w = gguf.GGUFWriter(file, "llama")
    for f in reader.fields.values():
        if not f.name.startswith("GGUF.") and f.name != "general.architecture":
            w.add_key_value(f.name, f.contents(), f.types[0], f.types[-1] if len(f.types) > 1 else None)
    for t in reader.tensors:
        values = gguf.quants.dequantize(t.data, t.tensor_type)
        if values.ndim == 2:
            w.add_tensor(t.name, gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(t.name, values)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    distill = Distill(m, write_text(tmp_path / "t.txt"))
    net = distill.student(file)
    q = net.model.layers[0].self_attn.q_proj
    theta = q.parametrizations.weight[0].theta
    theta.data = torch.linspace(-0.3, 0.3, theta.numel(), device=theta.device).view(theta.shape)
    out = distill.write(file, tmp_path / "out.gguf", net)
    written = unpermute(decoded(out)["blk.0.attn_q.weight"], HEADS)  # back in transformers' order
    trained = q.weight.detach().cpu().numpy()
    assert np.all(np.abs(written - trained) <= 2**-8 * np.abs(trained).max(axis=1, keepdims=True))
    assert KV_HEADS != HEADS  # (k is reordered by the KV heads: covered by the permute test)


# --- training ---

def test_the_loss_is_the_exact_kl_of_the_top_64_and_the_rest() -> None:
    torch.manual_seed(0)
    p, q = torch.log_softmax(torch.randn(3, 300), -1), torch.log_softmax(torch.randn(3, 300), -1)
    top = torch.topk(p, 64, -1)

    def coarse(logp: torch.Tensor) -> torch.Tensor:  # the 64 words, then everything else as one
        words = logp.gather(-1, top.indices)
        return torch.cat([words, torch.log1p(-words.exp().sum(-1, keepdim=True))], -1)
    want = (coarse(p).exp() * (coarse(p) - coarse(q))).sum(-1).mean()
    assert torch.allclose(kl_top(top.values, q.gather(-1, top.indices)), want, atol=1e-5)
    assert kl_top(top.values, top.values) == pytest.approx(0, abs=1e-6)  # the teacher itself: no loss


def llama_tokens(m: Model, text: Path) -> np.ndarray:
    """text tokenized by llama.cpp itself (the same tokens llama-perplexity reads)."""
    out = subprocess.run(["llama-tokenize", "-m", m.original, "-f", text, "--ids", "--log-disable"], check=True,
                         capture_output=True, text=True).stdout
    return np.array(json.loads(out.strip().splitlines()[-1]))


@needs_perplexity
def test_training_lowers_the_drift_llama_cpp_measures(distill, m, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(distill_module, "STEPS", 80)
    monkeypatch.setattr(Model, "tokens", lambda self, text: llama_tokens(m, text))  # train on real text
    write_text(distill.text, words=6000, seed=1)
    file = quantized(m, "Q2_K")
    losses = distill.run(file, tmp_path / "distilled.gguf")
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    fitness = Fitness(m, write_text(tmp_path / "kl.txt", seed=3), {})
    assert fitness.kl(tmp_path / "distilled.gguf")[0] < fitness.kl(file)[0]


def test_each_sequence_is_used_once_before_any_repeats(distill: Distill, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(distill_module, "STEPS", 12)

    class Net(torch.nn.Module):  # records which sequences it is given
        config = type("C", (), {})()

        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.theta = torch.nn.Parameter(torch.zeros(1))

        def activation_checkpointing_enable(self, kwargs) -> None:
            pass

        def forward(self, ids):
            seen.extend(ids[:, 0].tolist())
            logits = torch.zeros(*ids.shape, 300, device=ids.device) + self.theta
            return type("O", (), {"logits": logits})()
    teacher = {"ids": torch.arange(10)[:, None].repeat(1, 4), "top_ids": torch.zeros(10, 4, 64, dtype=torch.int32),
               "top_logp": torch.zeros(10, 4, 64)}
    distill.train(Net().to(distill.device), teacher)
    assert sorted(seen[:10]) == list(range(10)) and len(seen) == 24
    assert ROLES  # (the module map is checked by the loading tests)
