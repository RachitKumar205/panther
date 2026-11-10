"""Search baselines that use the same price tags as the solver (PLAN.md §12)."""

from __future__ import annotations

import random

from panther.search.problem import Problem


def greedy_per_byte(problem: Problem) -> dict[str, str] | None:
    """B4: start every gene at its largest format; repeatedly take the downgrade with the smallest
    penalty increase per byte saved until the budget is met (HAWQ-style)."""
    order = {g: sorted(fs, key=lambda f: problem.bytes_[g][f], reverse=True) for g, fs in problem.formats.items()}
    a = {g: fs[0] for g, fs in order.items()}
    while not problem.feasible(a):
        best = None
        for g, fs in order.items():
            i = fs.index(a[g])
            if i + 1 >= len(fs):
                continue
            nxt = fs[i + 1]
            saved = problem.bytes_[g][a[g]] - problem.bytes_[g][nxt]
            if saved <= 0:
                continue
            cost = (problem.penalty[g][nxt] - problem.penalty[g][a[g]]) / saved
            if best is None or cost < best[0]:
                best = (cost, g, nxt)
        if best is None:
            return None  # cannot fit even at the smallest formats
        a[best[1]] = best[2]
    return a


def random_feasible(problem: Problem, n: int, seed: int = 0, max_tries: int = 100_000) -> list[dict[str, str]]:
    """B5 candidate generator: random assignments that satisfy the budget. Scored by the caller with
    the same number of server quality evaluations as the solver pipeline."""
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    seen: set[tuple] = set()
    for _ in range(max_tries):
        a = {g: rng.choice(fs) for g, fs in problem.formats.items()}
        key = tuple(sorted(a.items()))
        if key in seen or not problem.feasible(a):
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= n:
            break
    return out
