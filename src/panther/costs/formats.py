"""Byte sizes of GGUF weight formats, taken from gguf-py so they match llama.cpp exactly."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from panther.models.inventory import CONFIG_DIR, Inventory, TensorSpec


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
    with open(Path(path) if path else CONFIG_DIR / "formats.yaml") as f:
        return yaml.safe_load(f)


def candidate_formats() -> list[str]:
    return list(load_format_config()["formats"])


# Genes without importance-matrix data: the token embedding (a lookup) and the untied output head
# (llama-imatrix skips the output tensor by default at the pinned commit). D21.
NO_IMATRIX_GENE_KINDS = frozenset({"embd", "embd_head", "head"})


def head_has_imatrix(model: str) -> bool:
    """True if the model's importance matrix was collected with --process-output (configs/models.yaml), so the
    untied output head may use imatrix-only formats (D46)."""
    from panther.models.inventory import load_registry
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
