"""The pipeline's decisions that need no server: types per budget, the RAM window, the baseline, part costs."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    # the start type re-measured (KL noise around 0) must not create a second point at the start: costs stay finite
    table2 = {b: {**row, p.base_format(M15): 0.002} for b, row in table.items()}
    pen2 = p.penalties(M15, table2, err)
    assert all(math.isfinite(v) for d in pen2.values() for v in d.values())
    assert all(pen2[g][start[g]] == 0 for g in pen2)


def test_probe_file_matches_normal_assembly_and_llama_layout(tmp_path, monkeypatch):
    """A probe file built by copying ranges holds exactly the tensors of a normally assembled file, in llama.cpp's
    required order (each tensor right after the previous one, aligned)."""
    import gguf
    import numpy as np
    from test_gguf import TINY, _write_source

    from panther.gguf_files import TensorStore, assemble
    from panther.model import build_matrix_inventory
    inv = build_matrix_inventory(TINY)
    (tmp_path / "store" / "tiny").mkdir(parents=True)
    srcs = {f: _write_source(inv, f, tmp_path / "store" / "tiny" / f"tiny-{f}.gguf", seed=i)
            for i, f in enumerate(["Q8_0", "Q4_0"])}
    monkeypatch.setattr(p, "DATA", tmp_path)
    monkeypatch.setattr(p, "inv_of", lambda m: inv)
    monkeypatch.setattr(p, "base_format", lambda m: "Q8_0")
    monkeypatch.setattr(p, "fill", lambda m, f: {g.gene_id: f for g in inv.genes})
    monkeypatch.setattr(p, "store", lambda m: TensorStore(srcs))
    monkeypatch.setattr(p, "out", lambda *parts: (tmp_path / "rw").joinpath(*parts) if (tmp_path / "rw").joinpath(
        *parts).parent.mkdir(parents=True, exist_ok=True) is None else None)
    a = {g.gene_id: ("Q4_0" if g.layer == 5 or g.kind == "head" else "Q8_0") for g in inv.genes}
    fast = p.fast_probe("tiny", a)
    slow = tmp_path / "slow.gguf"
    assemble(inv, a, TensorStore(srcs), slow, base_format="Q8_0")
    rf, rs = ({t.name: t for t in gguf.GGUFReader(x).tensors} for x in (fast, slow))
    assert rf.keys() == rs.keys()
    assert all(rf[n].tensor_type == rs[n].tensor_type and np.array_equal(rf[n].data, rs[n].data) for n in rs)


def test_one_genotype_gets_every_test_and_counts_as_done(tmp_path, monkeypatch):
    """run_all_tests calls every quality test once; tests_done then sees all of them (fake llama.cpp)."""
    import json
    calls = []

    def fake_run(kind, model, gguf, label, threads, reference=None, text=None, task=None, split=None):
        calls.append(kind if kind != "task" else task)
        d = tmp_path / "results" / model
        d.mkdir(parents=True, exist_ok=True)
        how = {"build": "cuda", "reference": reference, "text": text, "task": task, "split": split}
        (d / f"{len(calls)}.json").write_text(json.dumps({"eval_id": f"{len(calls):04d}", "label": label, "ok": True,
                                                         "how": {k: v for k, v in how.items() if v}, "metrics": {}}))
        return True

    monkeypatch.setattr(p, "DATA", tmp_path)
    monkeypatch.setattr(p, "RW", tmp_path / "rw")
    monkeypatch.setattr(p.evaluate, "run", fake_run)
    monkeypatch.setattr(p, "gsm8k_run", lambda m, path, label: p.out(m, "gsm8k", f"{label.replace(':', '_')}.json")
                        .write_text("{}"))
    assert not p.tests_done("m", "rw:b7:x", with_kl=True)
    p.run_all_tests("m", tmp_path / "f.gguf", "rw:b7:x", with_kl=True)
    assert sorted(calls) == sorted(["kl", "mmlu", "hellaswag", "arc_challenge", "winogrande", "ppl"])
    assert p.tests_done("m", "rw:b7:x", with_kl=True)


def test_accuracy_tie_goes_to_lower_kl():
    """A 0.1-point accuracy lead is noise on a few hundred questions: the lower KL wins. A 10-point lead is not."""
    import numpy as np
    from panther.stats import Evaluation
    rng = np.random.default_rng(1)
    base = rng.random(600) < 0.75

    def ev(name, flips):
        c = base.copy()
        c[:flips] = ~c[:flips]
        return Evaluation(name, {"t": c.astype(np.int8)}, np.ones(4), np.ones(4), 0, 0.0)
    evs = {"a": ev("a", 0), "b": ev("b", 2), "c": ev("c", 60)}
    rows = {n: {"mean_accuracy_pct": 100 * e.mean_accuracy(), "kl": kl}
            for (n, e), kl in zip(evs.items(), [0.14, 0.07, 0.01])}
    win, tied = p.choose_winner({n: rows[n] for n in "ab"}, evs)
    assert (win, sorted(tied)) == ("b", ["a", "b"])
    top = max("ac", key=lambda n: rows[n]["mean_accuracy_pct"])
    assert p.choose_winner({n: rows[n] for n in "ac"}, evs)[0] == top


@pytest.mark.parametrize("b,model,start", [(7, M14, "IQ2_S"), (1, M15, "IQ2_XXS")])
def test_padded_preset_layout_is_inside_the_window_and_only_raises(budget, b, model, start):
    budget(b)
    a = p.fill(model, start)
    a[next(iter(a))] = "Q4_K"  # a mixed start, like a preset layout
    padded = p.padded_from(model, a, b)
    assert padded is not None and p.in_window(model, padded, b)
    bpw = p.bits_per_weight
    assert all(bpw(padded[g]) >= bpw(a[g]) for g in a)
