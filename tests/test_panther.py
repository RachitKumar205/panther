"""Checks that need no GPU: llama.cpp output parsing, the RAM model, the GA's operators, the relative score."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from panther import measure as ms
from panther import model, search
from panther.model import GIB, Model, bpw

FIX = Path(__file__).parent / "fixtures" / "llama"
MODEL = "Qwen2.5-7B-Instruct"


def test_parsers():
    kl = ms.parse_kl((FIX / "kl_q6k_48chunks.log").read_text())
    assert 0 < kl["kl_mean"] < 0.1 and 90 < kl["top1_pct"] <= 100
    ch = kl["kl_chunks"]  # recovered from the running means
    assert len(ch) == 48 and abs(np.mean(ch) - kl["kl_mean"]) < 1e-5
    assert abs(ch[0] - 0.00634) < 1e-9 and abs(ch[1] - (2 * 0.00578 - 0.00634)) < 1e-9
    with pytest.raises(ms.ParseError):  # a chunk line missing
        ms.parse_kl(re.sub(r"(?m)^  47 .*\n", "", (FIX / "kl_q6k_48chunks.log").read_text()))
    assert ms.parse_ppl((FIX / "ppl_type1_q8_5chunks.log").read_text()) > 1
    text = "\n".join(f"{i}\t{100 * c / i:.8f}" for i, c in enumerate([1, 1, 2, 3], 1))
    assert ms.parse_choice(text, 4).tolist() == [1, 0, 1, 1]
    with pytest.raises(ms.ParseError):
        ms.parse_choice(text, 5)


def test_ram_model():
    m = Model(MODEL)
    assert len(m.matrices) == 2 + 7 * m.layers  # untied: embedding and head
    a = {x.name: "Q4_K" for x in m.matrices}
    assert 4 * GIB < m.ram(a) < 6 * GIB
    assert bpw("Q4_K") == 4.5 and len(m.allowed()["blk.0.attn_q.weight"]) == 14
    assert all(t not in m.allowed()["token_embd.weight"] for t in ("IQ1_S", "IQ2_XS"))


@pytest.fixture
def genes(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "RW", tmp_path)
    m = Model(MODEL)
    err = {x.name: {t: (x.rows * x.cols) / bpw(t) ** 2 for t in m.allowed()[x.name]} for x in m.matrices}
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / "error.json").write_text(json.dumps({"inputs": "", "error": err}))
    return search.Genes(MODEL, 5)


def test_ga_operators_stay_in_the_window(genes):
    pop = genes.initial(None)
    assert len(pop) >= 8 and all(genes.fits(a) for a in pop)
    for _ in range(20):
        assert genes.fits(genes.mutate(pop[0])) and genes.fits(genes.crossover(pop[0], pop[1]))


def test_solver_proposals(genes):
    pop = genes.initial(None)
    scored = [(sum(genes.err[g][a[g]] for g in a) / 1e9, a) for a in pop]
    assert all(genes.fits(a) for a in genes.proposals(scored, 2))


def test_relative():
    rng = np.random.default_rng(1)
    o = {"t": (rng.random(2000) < 0.7).astype(np.int8)}
    a = {"t": np.where(rng.random(2000) < 0.05, 1 - o["t"], o["t"])}  # ~5% flips
    b = {"t": np.where(rng.random(2000) < 0.10, 1 - o["t"], o["t"])}  # ~10% flips
    ca, cb = list(rng.normal(0.1, 0.02, 49)), list(rng.normal(0.2, 0.02, 49))
    rel, lo, hi = ms.relative(a, b, o, 0.1, ca, 0.2, cb, 10)
    assert 40 < rel < 60 and hi < 100 and ms.verdict(lo, hi) == "WIN"
    assert ms.verdict(*ms.relative(b, a, o, 0.2, cb, 0.1, ca, 10)[1:]) == "LOSE"
    assert ms.verdict(*ms.relative(a, a, o, 0.1, ca, 0.1, ca, 10)[1:]) == "TIE"
    with pytest.raises(ValueError):  # chunks of another text (numpy refuses the pairing)
        ms.kl_se(ca, cb[:-1])
    assert ms.flips(a, a) == 0


def test_kl_se_is_paired_over_chunks():
    """Two files' KL on the same chunks rise and fall together: the paired SE is far below the independent one."""
    rng = np.random.default_rng(2)
    hard = rng.gamma(2, 0.05, 110)  # how hard each chunk of text is, shared
    a, b = hard * 1.0 + rng.normal(0, 0.002, 110), hard * 1.1 + rng.normal(0, 0.002, 110)
    independent = np.hypot(np.std(a, ddof=1), np.std(b, ddof=1)) / np.sqrt(110)
    assert ms.kl_se(list(a), list(b)) < independent / 3


def test_reorder_interleaves_each_heads_halves():
    """llama.cpp's q/k row order (the distillation writes scales back in it): in each head of d rows, row 2i is
    row i and row 2i+1 is row i + d/2."""
    from panther.distill import reorder
    r = reorder(np.arange(64)[:, None], 8)[:, 0]
    assert r[:8].tolist() == [0, 4, 1, 5, 2, 6, 3, 7] and r[8:16].tolist() == [8, 12, 9, 13, 10, 14, 11, 15]


def test_a_changed_constant_changes_the_inputs(monkeypatch):
    for m in (ms, search):
        monkeypatch.setattr(m, "sha", lambda f: "0" * 64)
        monkeypatch.setattr(m, "llama_commit", lambda: "a" * 40)
    base = search.preset_ins(MODEL, 5)  # the GA's inputs carry the preset's
    monkeypatch.setattr(ms, "KL_WEIGHT", 20)
    assert search.preset_ins(MODEL, 5) != base
    monkeypatch.setattr(ms, "KL_WEIGHT", 10)
    assert search.preset_ins(MODEL, 5) == base
    monkeypatch.setattr(search, "WINDOW", (-0.02, 0.005))
    assert search.preset_ins(MODEL, 5) != base


def test_offline_error_weighting():
    """Columns weighted by their mean squared input, the word table's rows by their tokens' frequency; a tied word
    table both; weights that do not fit the matrix are an error."""
    d2 = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
    cols, rows = np.array([0.5, 2.0]), np.array([0.2, 0.3, 0.5])
    assert search.weighted(d2, cols, None) == 0.5 * 4 + 2.0 * 7
    assert search.weighted(d2, None, rows) == pytest.approx(0.2 * 3 + 0.3 * 7 + 0.5 * 1)
    assert search.weighted(d2, cols, rows) == pytest.approx(16 + 3.2)
    for bad in ((np.ones(3), None), (None, np.ones(2))):
        with pytest.raises(ValueError):
            search.weighted(d2, *bad)
