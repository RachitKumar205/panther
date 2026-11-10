import numpy as np

from panther.stats.decision import Evaluation, compare, holm

TASKS = {"hellaswag": 2000, "arc": 1000}


def _eval(name, accs, nll_per_tok=2.0, ram=100, tps=1.0, seed=0):
    rng = np.random.default_rng(seed)
    correct = {}
    for (t, n), p in zip(TASKS.items(), accs):
        correct[t] = (rng.random(n) < p).astype(np.int8)
    toks = np.full(400, 512)
    nll = rng.normal(nll_per_tok, 0.05, 400) * toks
    return Evaluation(name, correct, nll, toks, ram, tps)


def test_clear_accuracy_win():
    a = _eval("A", [0.80, 0.70], seed=1)
    b = _eval("B", [0.70, 0.60], seed=2)
    r = compare(a, b)
    assert r.winner == "A" and r.decided_by == "accuracy"


def test_identical_falls_through_to_ram():
    a = _eval("A", [0.75, 0.65], seed=3, ram=90)
    b = Evaluation("B", a.task_correct, a.chunk_nll, a.chunk_tokens, 100, 1.0)
    r = compare(a, b)
    assert r.decided_by == "ram" and r.winner == "A"


def test_perplexity_tiebreak():
    a = _eval("A", [0.75, 0.65], seed=4)
    b = Evaluation("B", a.task_correct, a.chunk_nll * 1.05, a.chunk_tokens, a.peak_ram_bytes, 1.0)
    r = compare(a, b)
    assert r.decided_by == "perplexity" and r.winner == "A"


def test_holm():
    assert holm({"x": 0.001, "y": 0.02, "z": 0.04}) == {"x": True, "y": True, "z": True}
    assert holm({"x": 0.001, "y": 0.04, "z": 0.04}) == {"x": True, "y": False, "z": False}
