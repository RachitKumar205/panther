"""Exact solver (multiple-choice knapsack, CP-SAT) with top-k enumeration by no-good cuts.

Optimal only with respect to the price tags: penalties are assumed additive across genes. The
shortlist (PLAN.md §8 step 3) exists because they are not, exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from panther.search.problem import Problem

PENALTY_SCALE = 1_000_000  # CP-SAT needs integers; penalties are rounded to 1e-6
SPEED_SCALE = 1_000_000_000  # seconds -> nanoseconds


@dataclass(frozen=True)
class Solution:
    assignment: dict[str, str]
    penalty: float
    weight_bytes: int


def solve_top_k(problem: Problem, k: int = 1, time_limit_s: float = 30.0) -> list[Solution]:
    """Return up to k distinct optimal-then-next-best assignments, best first."""
    m = cp_model.CpModel()
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for g, fmts in problem.formats.items():
        for f in fmts:
            x[g, f] = m.NewBoolVar(f"x[{g},{f}]")
        m.AddExactlyOne(x[g, f] for f in fmts)

    m.Add(
        sum(problem.bytes_[g][f] * v for (g, f), v in x.items()) <= problem.weight_budget_bytes
    )

    if problem.s_min is not None:
        sp = problem.speed
        assert sp is not None
        read = {gene.gene_id for gene in problem.inv.genes if gene.read_per_token}
        budget_ns = int((1.0 / problem.s_min - sp.overhead_s) * SPEED_SCALE)
        m.Add(
            sum(
                math.ceil(sp.seconds_per_byte[f] * problem.bytes_[g][f] * SPEED_SCALE) * v  # round up: conservative
                for (g, f), v in x.items()
                if g in read
            )
            <= budget_ns
        )

    m.Minimize(sum(round(problem.penalty[g][f] * PENALTY_SCALE) * v for (g, f), v in x.items()))

    solutions: list[Solution] = []
    n_genes = len(problem.formats)
    for _ in range(k):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_workers = 8
        status = solver.Solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        a = {g: f for (g, f), v in x.items() if solver.Value(v)}
        if not problem.feasible(a):  # guard against integer rounding of the price tags
            raise RuntimeError("solver returned an assignment that fails the exact feasibility check")
        solutions.append(Solution(a, problem.total_penalty(a), problem.total_bytes(a)))
        # No-good cut: forbid this exact assignment.
        m.Add(sum(x[g, f] for g, f in a.items()) <= n_genes - 1)
    return solutions
