"""The pipeline's decisions that need no server: types per budget, the RAM window, the baseline, part costs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import pipeline as p  # noqa: E402

M15, M14, M32 = "Qwen2.5-1.5B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"


@pytest.fixture
def budget():
    def set_(b):
        p.BUDGET = b
        p.menu.cache_clear()
        p.allowed.cache_clear()
    yield set_
    set_(None)


@pytest.mark.parametrize("b,model,start,probe,n", [(7, M14, "IQ3_S", "Q5_K", 7), (7, M32, "IQ1_S", "IQ3_S", 7),
                                                   (1, M15, "IQ3_XXS", "Q5_K", 10)])
def test_types_follow_the_budget(budget, b, model, start, probe, n):
    budget(b)
    assert (p.base_format(model), p.probe_format(model), len(p.menu(model))) == (start, probe, n)


def test_every_step_needs_a_budget(budget):
    budget(None)
    with pytest.raises(SystemExit):
        p.menu(M14)


@pytest.mark.parametrize("b,model", [(7, M14), (7, M32), (1, M15)])
def test_padded_baseline_is_inside_the_window(budget, b, model):
    budget(b)
    u, a = p.padded_uniform(model, b)
    assert p.in_window(model, a, b)
    assert not p.in_window(model, p.fill(model, p.menu(model)[0]), b * 1.2)  # far below a bigger budget


def test_part_costs_reproduce_measurements_and_are_zero_at_the_start(budget):
    budget(1)
    inv = p.inv_of(M15)
    err = {g.tensors[0].name: {f: g.n_weights * 2.0 ** -p.bits_per_weight(f) for f in p.allowed(M15)[g.gene_id]}
           for g in inv.genes}
    probe = p.probe_format(M15)
    table = {b: {probe: -0.01 * (i + 1)} for i, b in enumerate(p.blocks(M15))}
    pen = p.penalties(M15, table, err)
    start = p.fill(M15, p.base_format(M15))
    assert all(pen[g][start[g]] == 0 for g in pen)
    for b, row in table.items():  # a whole part moved to the probe costs exactly what was measured
        a = p.block_assignment(M15, b, probe)
        assert sum(pen[g][a[g]] for g in p.blocks(M15)[b]) == pytest.approx(row[probe])
