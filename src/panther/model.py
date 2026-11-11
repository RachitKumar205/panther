"""Model geometry, weight matrices and parts, quantization types, and the RAM model."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json

from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def build_matrix_inventory(geo: ModelGeometry) -> Inventory:
    """Every weight matrix its own gene (D46): attn_q.L, attn_k.L, attn_v.L, attn_output.L, ffn_gate.L,
    ffn_up.L, ffn_down.L, plus embd/head (or embd_head). Kinds and layers are those of the matrix's block, so
    format rules (gene_formats) and the RAM model apply unchanged."""
    blocks = build_inventory(geo)
    genes: list[Gene] = []
    for g in blocks.genes:
        if g.kind in ("attn", "mlp"):
            for t in g.tensors:
                short = t.name.split(".")[2]  # blk.L.<short>.weight
                genes.append(Gene(f"{short}.{g.layer}", g.kind, g.layer, (t,), read_per_token=g.read_per_token))
        else:
            genes.append(g)
    return Inventory(geo, tuple(genes), blocks.fixed_tensors)


def block_of(gene: Gene) -> str:
    """Sensitivity block of a (matrix) gene: attn.L, mlp.L, embd, head or embd_head (D45)."""
    return f"{gene.kind}.{gene.layer}" if gene.kind in ("attn", "mlp") else gene.kind


def qtype(name: str) -> GGMLQuantizationType:
    return GGMLQuantizationType[name]


def tensor_bytes(t: TensorSpec, fmt: str) -> int:
    """Exact on-disk / in-RAM bytes of one tensor stored in `fmt`."""
    block, block_bytes = GGML_QUANT_SIZES[qtype(fmt)]
    if t.cols % block != 0:
        raise ValueError(f"{t.name}: row width {t.cols} not a multiple of {fmt} block size {block}")
    return t.rows * (t.cols // block) * block_bytes


def bits_per_weight(fmt: str) -> float:
    block, block_bytes = GGML_QUANT_SIZES[qtype(fmt)]
    return 8.0 * block_bytes / block


@lru_cache(maxsize=1)
def load_format_config(path: str | None = None) -> dict:
    """configs/search.yaml: the quantization types and which of them need importance-matrix data."""
    with open(Path(path) if path else CONFIG_DIR / "search.yaml") as f:
        return yaml.safe_load(f)


def candidate_formats() -> list[str]:
    return list(load_format_config()["formats"])


# Genes without importance-matrix data: the token embedding (a lookup) and the untied output head
# (llama-imatrix skips the output tensor by default at the pinned commit). D21.
NO_IMATRIX_GENE_KINDS = frozenset({"embd", "embd_head", "head"})


def head_has_imatrix(model: str) -> bool:
    """True if the model's importance matrix was collected with --process-output (configs/models.yaml), so the
    untied output head may use imatrix-only formats (D46)."""
    m = next((m for m in load_registry()["models"] if m["name"] == model), {})
    return bool(m.get("imatrix_process_output"))


def gene_formats(inv: Inventory) -> dict[str, list[str]]:
    """Allowed formats per gene: all candidates, minus imatrix-only formats for genes without importance data."""
    needs = set(load_format_config()["requires_imatrix"])
    all_f = candidate_formats()
    no_imatrix = NO_IMATRIX_GENE_KINDS - ({"head"} if head_has_imatrix(inv.geometry.name) else set())
    return {
        g.gene_id: [f for f in all_f if not (g.kind in no_imatrix and f in needs)]
        for g in inv.genes
    }


def source_assignment(inv: Inventory, fmt: str, fallback: str = "Q2_K") -> dict[str, str]:
    """What the tensor-store source file for `fmt` holds: `fmt` everywhere it is allowed, `fallback`
    in genes where it is not (those bytes are never copied into a mixed file)."""
    allowed = gene_formats(inv)
    return {g: (fmt if fmt in fs else fallback) for g, fs in allowed.items()}


GIB = 1024**3


def gene_bytes(gene: Gene, fmt: str) -> int:
    return sum(tensor_bytes(t, fmt) for t in gene.tensors)


def fixed_bytes(inv: Inventory) -> int:
    """F32 norms and biases, which are never quantized."""
    return sum(t.n_elements * 4 for t in inv.fixed_tensors)


@dataclass(frozen=True)
class RamEstimate:
    weights: int
    kv_cache: int
    overhead: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.overhead

    @property
    def total_gib(self) -> float:
        return self.total / GIB


def estimate_ram(
    inv: Inventory,
    assignment: dict[str, str],
    context_tokens: int,
    overhead_gib: float,
    kv_bytes_per_element: float = 2.0,
) -> RamEstimate:
    """Predicted peak RAM for one configuration. `assignment` maps gene_id -> format."""
    missing = {g.gene_id for g in inv.genes} - assignment.keys()
    if missing:
        raise ValueError(f"assignment missing genes: {sorted(missing)[:5]}")
    weights = fixed_bytes(inv) + sum(gene_bytes(g, assignment[g.gene_id]) for g in inv.genes)
    return RamEstimate(
        weights=weights,
        kv_cache=inv.geometry.kv_cache_bytes(context_tokens, kv_bytes_per_element),
        overhead=int(overhead_gib * GIB),
    )
