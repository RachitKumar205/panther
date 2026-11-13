"""A model's weight matrices, their size in each quantization type, and the RAM model (weights + KV cache +
runtime overhead at the configured context)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

REPO = Path(__file__).resolve().parents[2]
GIB = 1024**3
CFG = yaml.safe_load((REPO / "configs/search.yaml").read_text())
REGISTRY = yaml.safe_load((REPO / "configs/models.yaml").read_text())
MODELS = {m["name"]: m for m in REGISTRY["models"]}
KV_BPE = {"Q8_0": 34 / 32, "F16": 2.0}[CFG["kv_cache_type"]]


def bpw(t: str) -> float:
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType[t]]
    return 8 * size / block


@dataclass(frozen=True)
class Matrix:
    name: str   # GGUF tensor name
    rows: int
    cols: int
    layer: int  # -1: token embedding / output head
    role: str   # attn_q ... ffn_down, token_embd, output

    def nbytes(self, t: str) -> int:
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[t]]
        return self.rows * (self.cols // block) * size


@lru_cache(maxsize=None)
class Model:
    def __init__(self, name: str) -> None:
        c = json.loads((REPO / "configs/model_configs" / f"{name}.json").read_text())
        self.name, self.layers, h = name, c["num_hidden_layers"], c["hidden_size"]
        hd = c.get("head_dim") or h // c["num_attention_heads"]
        q, kv, i = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd, c["intermediate_size"]
        self.kv_dim, tied = kv, bool(c.get("tie_word_embeddings", False))
        mats = [Matrix("token_embd.weight", c["vocab_size"], h, -1, "token_embd")]
        if not tied:
            mats.append(Matrix("output.weight", c["vocab_size"], h, -1, "output"))
        for L in range(self.layers):
            for role, r, k in (("attn_q", q, h), ("attn_k", kv, h), ("attn_v", kv, h), ("attn_output", h, q),
                               ("ffn_gate", i, h), ("ffn_up", i, h), ("ffn_down", h, i)):
                mats.append(Matrix(f"blk.{L}.{role}.weight", r, k, L, role))
        self.matrices = mats
        bias = (q + 2 * kv) if c.get("model_type") == "qwen2" else 0  # Qwen2: q/k/v biases (F32)
        self.fixed = 4 * (self.layers * (2 * h + bias) + h)  # F32 norms and biases
        no_imat = {"token_embd"} | (set() if MODELS[name].get("imatrix_process_output") else {"output"})
        self.no_imatrix = {m.name for m in mats if m.role in no_imat}

    def runtime(self) -> int:
        """RAM besides the weights: KV cache at the configured context, and llama.cpp's overhead."""
        return int(2 * self.layers * self.kv_dim * CFG["context_tokens"] * KV_BPE + CFG["overhead_gib"] * GIB)

    def ram(self, a: dict[str, str]) -> int:
        """Predicted peak RAM in bytes of the assignment {tensor name: type}."""
        return self.fixed + sum(m.nbytes(a[m.name]) for m in self.matrices) + self.runtime()

    def avg_bpw(self, budget: float) -> float:
        return 8 * (budget * GIB - self.runtime() - self.fixed) / sum(m.rows * m.cols for m in self.matrices)

    def menu(self, budget: float) -> list[str]:
        """Types from 1.5 bits/weight below the budget's average to 2.5 above, low to high."""
        avg = self.avg_bpw(budget)
        return sorted((t for t in CFG["formats"] if avg - 1.5 <= bpw(t) <= avg + 2.5), key=bpw)

    def allowed(self, budget: float) -> dict[str, list[str]]:
        menu, need = self.menu(budget), set(CFG["requires_imatrix"])
        return {m.name: [t for t in menu if not (m.name in self.no_imatrix and t in need)] for m in self.matrices}
