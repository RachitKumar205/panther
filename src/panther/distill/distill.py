"""Distillation: one multiplier per block of 256 weights, and every norm, trained so that a finalist's next-word
predictions match the original's on the distill set; then written back into a copy of the finalist's file (the
same size and speed: only block scales and norms change)."""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
from pathlib import Path

import gguf
import numpy as np
import torch
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM

from panther.model import Model

SEQUENCE, BATCH, TOP, STEPS = 512, 2, 64, 600  # 600 updates of 2 sequences of 512 tokens; the teacher's top 64
RATE_SCALES, RATE_NORMS = 2e-3, 1e-4
BLOCK = 256  # weights per multiplier: one block of the K and IQ types (8 blocks of Q8_0)
# Where each type keeps its block's 16-bit scale d (and dmin, for types with a minimum): byte offsets inside one
# stored block (ggml's block structs; checked in the 21 Nov audit). IQ1_M has no single scale: it is left as it is.
SCALE_AT = {"Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q8_0": (0,),
            "IQ1_S": (0,), "IQ2_XXS": (0,), "IQ2_XS": (0,), "IQ2_S": (0,), "IQ3_XXS": (0,), "IQ4_XS": (0,)}
# transformers module → GGUF name, per layer
MATRICES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
            "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
            "mlp.down_proj": "ffn_down"}
NORMS = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
GIB = 1024**3
HEADROOM = 8 * GIB  # GPU memory kept for activations and logits when deciding whether 32-bit weights fit


def permute(rows: np.ndarray, heads: int) -> np.ndarray:
    """The row order llama.cpp gives attention q and k of llama-style models (Mistral) when converting them
    (convert_hf_to_gguf's LlamaModel.permute; heads: the head count of q, the KV head count of k)."""
    return rows.reshape(heads, 2, rows.shape[0] // heads // 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


def unpermute(rows: np.ndarray, heads: int) -> np.ndarray:
    """The file's q/k rows back in transformers' order."""
    return rows.reshape(heads, rows.shape[0] // heads // 2, 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


def kl_top(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """KL(teacher ‖ student) over 65 outcomes: the teacher's 64 most likely next words, and everything else as one.
    Both are log-probabilities of those 64 words (teacher: as recorded, student: from its whole vocabulary); neither
    is renormalized, so the student is not pushed to put all its probability on those 64 words."""
    rest_teacher = (1 - teacher.exp().sum(-1)).clamp_min(1e-12)
    rest_student = (1 - student.exp().sum(-1)).clamp_min(1e-12)
    top = (teacher.exp() * (teacher - student)).sum(-1)
    return (top + rest_teacher * (rest_teacher.log() - rest_student.log())).mean()


class BlockScale(torch.nn.Module):
    """A matrix's weights × exp(θ), one θ (float32, starting at 0) per block of 256 weights of a row."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(weight.shape[0], weight.shape[1] // BLOCK, device=weight.device))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return (w.float().view(w.shape[0], -1, BLOCK) * torch.exp(self.theta)[..., None]).view(w.shape).to(w.dtype)


class NormShift(torch.nn.Module):
    """A norm's weights + a float32 shift (starting at 0), so small updates are not lost in a 16-bit weight."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.shift = torch.nn.Parameter(torch.zeros(weight.shape, device=weight.device))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return (w.float() + self.shift).to(w.dtype)


class Distill:
    """Distillation for one model on the distill set's text. The teacher is the original (16-bit, as published); its
    top 64 next words at every position are recorded once, in data/models/<model>/distill/, named by the text's
    content."""

    def __init__(self, model: Model, text: Path) -> None:
        self.model, self.text = model, text
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.record = model.dir / "distill" / f"{hashlib.sha256(text.read_bytes()).hexdigest()[:16]}.pt"

    def run(self, file: Path, out: Path) -> list[float]:
        """Distills the finalist in file into out (a copy with new block scales and norms). Returns the loss of every
        update."""
        teacher = self.teacher()
        student = self.student(file)
        losses = self.train(student, teacher)
        self.write(file, out, student)
        del student
        gc.collect()  # parametrized modules hold reference cycles: collect them, or their GPU memory stays held
        torch.cuda.empty_cache()
        return losses

    def teacher(self) -> dict[str, torch.Tensor]:
        """The original's top 64 next-word log-probabilities (float32) at every position of the text, cut into
        sequences of 512 tokens."""
        if self.record.exists():
            return torch.load(self.record, weights_only=True)
        ids = torch.from_numpy(self.model.tokens(self.text))
        ids = ids[: len(ids) // SEQUENCE * SEQUENCE].view(-1, SEQUENCE)
        net = AutoModelForCausalLM.from_pretrained(self.model.hf, dtype=torch.bfloat16, device_map=self.device)
        tops = []
        with torch.no_grad():
            for start in range(0, len(ids), BATCH):
                logits = net(ids[start:start + BATCH].to(self.device)).logits
                tops.append(torch.topk(torch.log_softmax(logits.float(), -1), TOP, -1))
        record = {"ids": ids, "top_ids": torch.cat([t.indices for t in tops]).int().cpu(),
                  "top_logp": torch.cat([t.values for t in tops]).cpu()}
        del net
        torch.cuda.empty_cache()
        self.record.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, self.record.with_suffix(".part"))
        os.replace(self.record.with_suffix(".part"), self.record)
        return record

    def student(self, file: Path) -> torch.nn.Module:
        """The finalist as a transformers model: every matrix and norm holds exactly the file's decoded values, in
        float32 if the model fits on the GPU that way (1.5B, 7B, 14B), else in bf16 (22B, 24B: each weight within
        about 0.2 %). Block scales (where the type has one) and norm shifts are attached, the only trainable parts."""
        exact = not torch.cuda.is_available() or 2 * self.model.original.stat().st_size + HEADROOM <= \
            torch.cuda.mem_get_info()[0]
        net = AutoModelForCausalLM.from_pretrained(self.model.hf, dtype=torch.float32 if exact else torch.bfloat16,
                                                   device_map=self.device)
        net.requires_grad_(False)
        reader = gguf.GGUFReader(file)
        tensors, heads = {t.name: t for t in reader.tensors}, self.reordered(reader)
        for name, module in self.modules(net).items():
            if name not in tensors:  # a tied model has no output layer: its word table is used both ways
                continue
            t = tensors[name]
            values = gguf.quants.dequantize(t.data, t.tensor_type)
            if name.split(".")[-2] in heads:
                values = unpermute(values, heads[name.split(".")[-2]])
            module.weight.data.copy_(torch.tensor(values).view(module.weight.shape))
            if t.tensor_type.name in SCALE_AT and name != "token_embd.weight":
                parametrize.register_parametrization(module, "weight", BlockScale(module.weight))
            elif name.endswith("norm.weight"):
                parametrize.register_parametrization(module, "weight", NormShift(module.weight))
        return net

    def train(self, net: torch.nn.Module, teacher: dict[str, torch.Tensor]) -> list[float]:
        """600 updates of 2 sequences, each sequence used once (in a fixed random order) before any is repeated.
        Loss: kl_top at every position. Adam: the block scales at 2e-3, the norm shifts at 1e-4."""
        scales = [p for n, p in net.named_parameters() if n.endswith("theta")]
        shifts = [p for n, p in net.named_parameters() if n.endswith("shift")]
        optimizer = torch.optim.Adam([{"params": scales, "lr": RATE_SCALES}, {"params": shifts, "lr": RATE_NORMS}])
        net.train()
        net.config.use_cache = False
        net.activation_checkpointing_enable({"use_reentrant": False})  # recompute activations instead of keeping them
        rng = np.random.default_rng(0)
        order = np.concatenate([rng.permutation(len(teacher["ids"])) for _ in range(-(-STEPS * BATCH //
                                                                                       len(teacher["ids"])))])
        losses = []
        for step in range(STEPS):
            batch = torch.from_numpy(order[step * BATCH:(step + 1) * BATCH])
            logp = torch.log_softmax(net(teacher["ids"][batch].to(self.device)).logits.float(), -1)
            loss = kl_top(teacher["top_logp"][batch].to(self.device),
                          torch.gather(logp, -1, teacher["top_ids"][batch].to(self.device).long()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        return losses

    def write(self, file: Path, out: Path, net: torch.nn.Module) -> Path:
        """A copy of file in which each trained block's scale d (and dmin) is d × exp(θ), rounded to its 16 bits,
        and each norm is its trained value. Nothing else changes."""
        part = out.with_name(out.name + ".part")
        shutil.copy(file, part)
        reader = gguf.GGUFReader(part, "r+")
        tensors, heads = {t.name: t for t in reader.tensors}, self.reordered(reader)
        for name, module in self.modules(net).items():
            if not parametrize.is_parametrized(module, "weight"):
                continue
            change = module.parametrizations.weight[0]
            t = tensors[name]
            if isinstance(change, NormShift):  # the file's own 32-bit norm + the shift
                t.data[...] = t.data + change.shift.detach().cpu().numpy()
                continue
            factor = torch.exp(change.theta).detach().cpu().numpy()  # rows x blocks of 256, transformers' row order
            if name.split(".")[-2] in heads:
                factor = permute(factor, heads[name.split(".")[-2]])
            weights_per_block, bytes_per_block = gguf.GGML_QUANT_SIZES[t.tensor_type]
            blocks = t.data.reshape(factor.shape[0], -1, bytes_per_block)
            factor = np.repeat(factor, BLOCK // weights_per_block, axis=1)  # Q8_0: 8 blocks per 256 weights
            for at in SCALE_AT[t.tensor_type.name]:
                d = blocks[:, :, at:at + 2].copy().view(np.float16)[..., 0]
                blocks[:, :, at:at + 2] = (d.astype(np.float32) * factor).astype(np.float16)[..., None].view(np.uint8)
        reader.data.flush()
        del reader, tensors
        os.replace(part, out)
        return out

    def reordered(self, reader: gguf.GGUFReader) -> dict[str, int]:
        """For a llama-style model (Mistral): the head counts its q and k rows are reordered by (none for Qwen2)."""
        if reader.fields["general.architecture"].contents() != "llama":
            return {}
        return {"attn_q": reader.fields["llama.attention.head_count"].contents(),
                "attn_k": reader.fields["llama.attention.head_count_kv"].contents()}

    def modules(self, net: torch.nn.Module) -> dict[str, torch.nn.Module]:
        """GGUF tensor name → the transformers module holding it."""
        names = {"token_embd.weight": net.model.embed_tokens, "output.weight": net.lm_head,
                 "output_norm.weight": net.model.norm}
        for i, layer in enumerate(net.model.layers):
            names |= {f"blk.{i}.{role}.weight": layer.get_submodule(module) for module, role in MATRICES.items()}
            names |= {f"blk.{i}.{role}.weight": layer.get_submodule(module) for module, role in NORMS.items()}
        return names
