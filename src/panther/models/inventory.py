"""Model geometry and the tensor/gene inventory used by the RAM model, solver and GGUF tools.

Tensor names follow llama.cpp's GGUF naming for the qwen2 architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

# K-quant and IQ formats pack 256 weights per super-block along a row.
SUPER_BLOCK = 256


@dataclass(frozen=True)
class ModelGeometry:
    name: str
    n_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    vocab: int
    tied: bool
    max_context: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def kv_cache_bytes(self, context_tokens: int, bytes_per_element: float = 2.0) -> int:
        """K and V caches for every layer at the given context (f16 by default)."""
        return int(2 * self.n_layers * self.kv_dim * context_tokens * bytes_per_element)


@dataclass(frozen=True)
class TensorSpec:
    """One tensor in the GGUF file. `shape` is (rows, cols); cols is the contiguous row width."""

    name: str
    rows: int
    cols: int
    quantizable: bool  # False for norms and biases, which stay F32

    @property
    def n_elements(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class Gene:
    """A group of tensors that always share one format.

    `read_per_token` is False only for an untied token embedding: decode gathers one row of it per
    token, so it costs RAM but almost no memory traffic.
    """

    gene_id: str
    kind: str  # "attn" | "mlp" | "embd" | "head" | "embd_head"
    layer: int | None
    tensors: tuple[TensorSpec, ...]
    read_per_token: bool = True

    @property
    def n_weights(self) -> int:
        return sum(t.n_elements for t in self.tensors)


@dataclass(frozen=True)
class Inventory:
    geometry: ModelGeometry
    genes: tuple[Gene, ...]
    fixed_tensors: tuple[TensorSpec, ...] = field(default=())  # F32 norms/biases, never searched

    def gene(self, gene_id: str) -> Gene:
        for g in self.genes:
            if g.gene_id == gene_id:
                return g
        raise KeyError(gene_id)


def load_registry(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def load_geometry(name: str, config_dir: Path | None = None) -> ModelGeometry:
    path = (config_dir or CONFIG_DIR / "model_configs") / f"{name}.json"
    c = json.loads(path.read_text())
    return ModelGeometry(
        name=name,
        n_layers=c["num_hidden_layers"],
        hidden=c["hidden_size"],
        intermediate=c["intermediate_size"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"],
        vocab=c["vocab_size"],
        tied=bool(c.get("tie_word_embeddings", False)),
        max_context=c.get("max_position_embeddings", 0),
    )


def build_inventory(geo: ModelGeometry) -> Inventory:
    h, i, kv = geo.hidden, geo.intermediate, geo.kv_dim
    genes: list[Gene] = []
    fixed: list[TensorSpec] = []

    embd = TensorSpec("token_embd.weight", geo.vocab, h, True)
    if geo.tied:
        # One tensor serves as embedding and output head, so it is read in full every token.
        genes.append(Gene("embd_head", "embd_head", None, (embd,), read_per_token=True))
    else:
        genes.append(Gene("embd", "embd", None, (embd,), read_per_token=False))
        genes.append(
            Gene("head", "head", None, (TensorSpec("output.weight", geo.vocab, h, True),))
        )

    for L in range(geo.n_layers):
        p = f"blk.{L}."
        genes.append(
            Gene(
                f"attn.{L}",
                "attn",
                L,
                (
                    TensorSpec(p + "attn_q.weight", h, h, True),
                    TensorSpec(p + "attn_k.weight", kv, h, True),
                    TensorSpec(p + "attn_v.weight", kv, h, True),
                    TensorSpec(p + "attn_output.weight", h, h, True),
                ),
            )
        )
        genes.append(
            Gene(
                f"mlp.{L}",
                "mlp",
                L,
                (
                    TensorSpec(p + "ffn_gate.weight", i, h, True),
                    TensorSpec(p + "ffn_up.weight", i, h, True),
                    TensorSpec(p + "ffn_down.weight", h, i, True),
                ),
            )
        )
        fixed += [
            TensorSpec(p + "attn_norm.weight", 1, h, False),
            TensorSpec(p + "ffn_norm.weight", 1, h, False),
            TensorSpec(p + "attn_q.bias", 1, h, False),
            TensorSpec(p + "attn_k.bias", 1, kv, False),
            TensorSpec(p + "attn_v.bias", 1, kv, False),
        ]
    fixed.append(TensorSpec("output_norm.weight", 1, h, False))
    return Inventory(geo, tuple(genes), tuple(fixed))


def check_super_block_divisibility(inv: Inventory) -> list[str]:
    """Return tensors whose row width is not a multiple of 256 (K/IQ formats would fall back)."""
    return [
        t.name
        for g in inv.genes
        for t in g.tensors
        if t.cols % SUPER_BLOCK != 0
    ]
