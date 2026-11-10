"""RAM price tags (PLAN.md §7): exact weight bytes + KV cache + runtime overhead."""

from __future__ import annotations

from dataclasses import dataclass

from panther.costs.formats import tensor_bytes
from panther.models.inventory import Gene, Inventory

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


def uniform(inv: Inventory, fmt: str) -> dict[str, str]:
    return {g.gene_id: fmt for g in inv.genes}


def average_bits_per_weight(inv: Inventory, assignment: dict[str, str]) -> float:
    bits = sum(8 * gene_bytes(g, assignment[g.gene_id]) for g in inv.genes)
    weights = sum(g.n_weights for g in inv.genes)
    return bits / weights
