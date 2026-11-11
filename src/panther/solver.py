"""The allocation problem (one quantization type per matrix under a RAM budget) and its CP-SAT solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from panther.model import GIB, Inventory, fixed_bytes, gene_bytes

PENALTY_SCALE = 1_000_000  # CP-SAT needs integers; penalties are rounded to 1e-6


@dataclass
class Problem:
    inv: Inventory
    formats: dict[str, list[str]]  # gene_id -> allowed types
    penalty: dict[str, dict[str, float]]  # gene_id -> type -> predicted KL increase
    r_max_gib: float
    context_tokens: int
    overhead_gib: float
    kv_bytes_per_element: float = 2.0  # f16; q8_0 is 34/32
    bytes_: dict[str, dict[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.bytes_ = {g.gene_id: {f: gene_bytes(g, f) for f in self.formats[g.gene_id]} for g in self.inv.genes}
        for g in self.inv.genes:
            missing = set(self.formats[g.gene_id]) - self.penalty[g.gene_id].keys()
            if missing:
                raise ValueError(f"{g.gene_id}: no penalty for {sorted(missing)}")

    @property
    def weight_budget_bytes(self) -> int:
        """Bytes left for the searched weights after KV cache, overhead and F32 tensors."""
        geo = self.inv.geometry
        return (int(self.r_max_gib * GIB) - geo.kv_cache_bytes(self.context_tokens, self.kv_bytes_per_element)
                - int(self.overhead_gib * GIB) - fixed_bytes(self.inv))

    def total_bytes(self, a: dict[str, str]) -> int:
        return sum(self.bytes_[g][f] for g, f in a.items())

    def total_penalty(self, a: dict[str, str]) -> float:
        return sum(self.penalty[g][f] for g, f in a.items())

    def feasible(self, a: dict[str, str]) -> bool:
        return self.total_bytes(a) <= self.weight_budget_bytes


@dataclass(frozen=True)
class Solution:
    assignment: dict[str, str]
    penalty: float
    weight_bytes: int


def solve_top_k(problem: Problem, k: int = 1, time_limit_s: float = 30.0, min_bytes: int | None = None,
                min_distance: int = 1) -> list[Solution]:
    """Up to k assignments, best first. `min_bytes` forces the weights to fill the budget from below;
    `min_distance` makes each one differ from every earlier one in at least that many genes."""
    m = cp_model.CpModel()
    x = {(g, f): m.NewBoolVar(f"x[{g},{f}]") for g, fs in problem.formats.items() for f in fs}
    for g, fs in problem.formats.items():
        m.AddExactlyOne(x[g, f] for f in fs)
    total = sum(problem.bytes_[g][f] * v for (g, f), v in x.items())
    m.Add(total <= problem.weight_budget_bytes)
    if min_bytes is not None:
        m.Add(total >= min_bytes)
    m.Minimize(sum(round(problem.penalty[g][f] * PENALTY_SCALE) * v for (g, f), v in x.items()))
    solutions: list[Solution] = []
    for _ in range(k):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_workers = 8
        if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        a = {g: f for (g, f), v in x.items() if solver.Value(v)}
        if not problem.feasible(a):  # guard against integer rounding
            raise RuntimeError("solver returned an assignment that fails the exact feasibility check")
        solutions.append(Solution(a, problem.total_penalty(a), problem.total_bytes(a)))
        m.Add(sum(x[g, f] for g, f in a.items()) <= len(problem.formats) - min_distance)  # differ from this one
    return solutions
