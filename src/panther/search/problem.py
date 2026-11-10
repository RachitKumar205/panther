"""The allocation problem shared by the solver and the baselines (PLAN.md §7, §8).

Choose one format per gene to minimize total quality penalty subject to a RAM budget and, optionally,
a predicted decode-speed floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from panther.costs.ram import GIB, fixed_bytes, gene_bytes
from panther.models.inventory import Inventory


@dataclass(frozen=True)
class SpeedModel:
    """Decode seconds per token = overhead_s + Σ seconds_per_byte[fmt] × bytes read in fmt."""

    overhead_s: float
    seconds_per_byte: dict[str, float]


@dataclass
class Problem:
    inv: Inventory
    formats: dict[str, list[str]]  # gene_id -> allowed formats
    penalty: dict[str, dict[str, float]]  # gene_id -> fmt -> quality penalty (e.g. KL increase)
    r_max_gib: float
    context_tokens: int
    overhead_gib: float
    s_min: float | None = None
    speed: SpeedModel | None = None
    bytes_: dict[str, dict[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.bytes_ = {
            g.gene_id: {f: gene_bytes(g, f) for f in self.formats[g.gene_id]} for g in self.inv.genes
        }
        for g in self.inv.genes:
            missing = set(self.formats[g.gene_id]) - self.penalty[g.gene_id].keys()
            if missing:
                raise ValueError(f"{g.gene_id}: no penalty for {sorted(missing)}")
        if self.s_min is not None and self.speed is None:
            raise ValueError("s_min set but no speed model")

    @property
    def weight_budget_bytes(self) -> int:
        """Bytes left for searchable weights after KV cache, overhead and F32 tensors."""
        geo = self.inv.geometry
        return (
            int(self.r_max_gib * GIB)
            - geo.kv_cache_bytes(self.context_tokens)
            - int(self.overhead_gib * GIB)
            - fixed_bytes(self.inv)
        )

    def total_bytes(self, a: dict[str, str]) -> int:
        return sum(self.bytes_[g][f] for g, f in a.items())

    def total_penalty(self, a: dict[str, str]) -> float:
        return sum(self.penalty[g][f] for g, f in a.items())

    def decode_seconds(self, a: dict[str, str]) -> float | None:
        if self.speed is None:
            return None
        s = self.speed.overhead_s
        for g in self.inv.genes:
            if g.read_per_token:
                f = a[g.gene_id]
                s += self.speed.seconds_per_byte[f] * self.bytes_[g.gene_id][f]
        return s

    def feasible(self, a: dict[str, str]) -> bool:
        if self.total_bytes(a) > self.weight_budget_bytes:
            return False
        if self.s_min is not None:
            secs = self.decode_seconds(a)
            assert secs is not None
            return secs <= 1.0 / self.s_min
        return True
