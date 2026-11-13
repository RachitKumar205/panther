"""CP-SAT: one type per matrix, total bytes inside [lo, hi], least total penalty; k distinct answers."""

from __future__ import annotations

from ortools.sat.python import cp_model


def solve(cost: dict[str, dict[str, int]], penalty: dict[str, dict[str, float]], lo: int, hi: int, k: int = 1,
          min_distance: int = 3, seconds: float = 20) -> list[dict[str, str]]:
    """cost[matrix][type] = bytes; penalty[matrix][type] = predicted loss. Each answer differs from the earlier ones
    in at least `min_distance` matrices."""
    m = cp_model.CpModel()
    x = {(g, t): m.NewBoolVar(f"{g}:{t}") for g in cost for t in cost[g]}
    for g in cost:
        m.AddExactlyOne(x[g, t] for t in cost[g])
    total = sum(cost[g][t] * v for (g, t), v in x.items())
    m.Add(total <= hi)
    m.Add(total >= lo)
    m.Minimize(sum(round(1e6 * penalty[g][t]) * v for (g, t), v in x.items()))
    out = []
    for _ in range(k):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds, s.parameters.num_workers = seconds, 8
        if s.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        a = {g: t for (g, t), v in x.items() if s.Value(v)}
        out.append(a)
        m.Add(sum(x[g, t] for g, t in a.items()) <= len(cost) - min_distance)
    return out
