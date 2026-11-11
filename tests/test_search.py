import itertools

import pytest

from panther.costs.ram import GIB
from panther.models.inventory import ModelGeometry, build_inventory, load_geometry
from panther.search.problem import Problem, SpeedModel
from panther.search.solver import solve_top_k

FMTS = ["Q2_K", "Q3_K", "Q4_K"]
BASE_PEN = {"Q2_K": 1.0, "Q3_K": 0.3, "Q4_K": 0.1}

SMALL = ModelGeometry("small", n_layers=2, hidden=256, intermediate=512, n_heads=4,
                      n_kv_heads=2, vocab=256, tied=True, max_context=128)


def _r_max_at(inv, frac, sens=None):
    """R_max putting the weight budget a fraction `frac` of the way from all-smallest to all-largest."""
    probe = _problem(inv, 100.0, sens)
    fixed_and_kv = 100.0 * GIB - probe.weight_budget_bytes
    lo = sum(min(probe.bytes_[g].values()) for g in probe.formats)
    hi = sum(max(probe.bytes_[g].values()) for g in probe.formats)
    return (fixed_and_kv + lo + frac * (hi - lo)) / GIB


def _problem(inv, r_max_gib, sens=None, **kw):
    sens = sens or {}
    return Problem(
        inv=inv,
        formats={g.gene_id: list(FMTS) for g in inv.genes},
        penalty={g.gene_id: {f: BASE_PEN[f] * sens.get(g.gene_id, 1.0) for f in FMTS} for g in inv.genes},
        r_max_gib=r_max_gib, context_tokens=128, overhead_gib=0.0, **kw,
    )


def _brute_force(p):
    genes = list(p.formats)
    best = None
    for combo in itertools.product(*(p.formats[g] for g in genes)):
        a = dict(zip(genes, combo))
        if p.feasible(a) and (best is None or p.total_penalty(a) < best[0] - 1e-12):
            best = (p.total_penalty(a), a)
    return best


@pytest.mark.parametrize("frac", [0.35, 0.45, 0.6])
def test_solver_matches_brute_force(frac):
    inv = build_inventory(SMALL)
    sens = {"attn.0": 3.0, "mlp.1": 0.2, "embd_head": 5.0}
    r_max = _r_max_at(inv, frac, sens)
    p = _problem(inv, r_max, sens)
    sol = solve_top_k(p, k=1)[0]
    bf = _brute_force(p)
    assert bf is not None
    assert sol.penalty == pytest.approx(bf[0], abs=1e-5)
    assert p.feasible(sol.assignment)


def test_top_k_distinct_and_ordered():
    inv = build_inventory(SMALL)
    p = _problem(inv, _r_max_at(inv, 0.6, {"attn.0": 2.0}), {"attn.0": 2.0})
    sols = solve_top_k(p, k=5)
    assert len(sols) == 5
    assert len({tuple(sorted(s.assignment.items())) for s in sols}) == 5
    pens = [s.penalty for s in sols]
    assert pens == sorted(pens)


def test_speed_floor_binds():
    inv = build_inventory(SMALL)
    spb = {"Q2_K": 1e-9, "Q3_K": 3e-9, "Q4_K": 3e-9}  # Q3_K/Q4_K slow per byte
    p0 = _problem(inv, 100.0, speed=SpeedModel(0.0, spb))
    fast = solve_top_k(p0, k=1)[0]
    assert set(fast.assignment.values()) == {"Q4_K"}  # no floor: best quality
    secs = p0.decode_seconds({g: "Q2_K" for g in p0.formats})
    p1 = _problem(inv, 100.0, speed=SpeedModel(0.0, spb), s_min=1.0 / (secs * 1.5))
    sol = solve_top_k(p1, k=1)[0]
    assert p1.feasible(sol.assignment)
    assert p1.decode_seconds(sol.assignment) <= secs * 1.5 + 1e-12


def test_fill_window_and_distance():
    inv = build_inventory(SMALL)
    p = _problem(inv, _r_max_at(inv, 0.6))
    sols = solve_top_k(p, k=3, min_bytes=int(0.95 * p.weight_budget_bytes), min_distance=2)
    for i, s in enumerate(sols):
        assert s.weight_bytes >= int(0.95 * p.weight_budget_bytes)
        for t in sols[:i]:
            assert sum(s.assignment[g] != t.assignment[g] for g in s.assignment) >= 2


def test_14b_solves_quickly():
    inv = build_inventory(load_geometry("Qwen2.5-14B-Instruct"))
    p = Problem(
        inv=inv,
        formats={g.gene_id: list(FMTS) for g in inv.genes},
        penalty={g.gene_id: {f: BASE_PEN[f] * (1 + (hash(g.gene_id) % 7)) for f in FMTS} for g in inv.genes},
        r_max_gib=7.0, context_tokens=4096, overhead_gib=0.4,
    )
    sols = solve_top_k(p, k=3, time_limit_s=20)
    assert len(sols) == 3 and all(p.feasible(s.assignment) for s in sols)
