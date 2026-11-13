"""Checks that need no GPU: llama.cpp output parsing, the RAM model, the GA's operators, the bootstrap."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import pipeline as p  # noqa: E402
from panther import llama, stats  # noqa: E402
from panther.model import GIB, Model, bpw  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "llama"
MODEL = "Qwen2.5-7B-Instruct"


def test_parsers():
    kl = llama.parse_kl((FIX / "kl_q6k_48chunks.log").read_text())
    assert 0 < kl["kl_mean"] < 0.1 and 90 < kl["top1_pct"] <= 100
    assert llama.parse_ppl((FIX / "ppl_type1_q8_5chunks.log").read_text()) > 1
    text = "\n".join(f"{i}\t{100 * c / i:.8f}" for i, c in enumerate([1, 1, 2, 3], 1))
    assert llama.parse_choice(text, 4).tolist() == [1, 0, 1, 1]
    with pytest.raises(llama.ParseError):
        llama.parse_choice(text, 5)


def test_ram_model():
    m = Model(MODEL)
    assert len(m.matrices) == 2 + 7 * m.layers  # untied: embedding and head
    a = {x.name: "Q4_K" for x in m.matrices}
    assert 4 * GIB < m.ram(a) < 6 * GIB
    assert bpw("Q4_K") == 4.5 and len(m.allowed()["blk.0.attn_q.weight"]) == 14
    assert all(t not in m.allowed()["token_embd.weight"] for t in ("IQ1_S", "IQ2_XS"))


@pytest.fixture
def genes(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "RW", tmp_path)
    m = Model(MODEL)
    err = {x.name: {t: (x.rows * x.cols) / bpw(t) ** 2 for t in m.allowed()[x.name]} for x in m.matrices}
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / "error.json").write_text(json.dumps(err))
    return p.Genes(MODEL, 5)


def test_ga_operators_stay_in_the_window(genes):
    pop = genes.initial(None)
    assert len(pop) >= 8 and all(genes.fits(a) for a in pop)
    for _ in range(20):
        assert genes.fits(genes.mutate(pop[0])) and genes.fits(genes.crossover(pop[0], pop[1]))


def test_solver_proposals(genes):
    pop = genes.initial(None)
    scored = [(sum(genes.err[g][a[g]] for g in a) / 1e9, a) for a in pop]
    assert all(genes.fits(a) for a in genes.proposals(scored, 2))


def test_bootstrap():
    rng = np.random.default_rng(1)
    a = {"t": (rng.random(2000) < 0.7).astype(np.int8)}
    b = {"t": (rng.random(2000) < 0.6).astype(np.int8)}
    d, lo, hi = stats.compare(a, b)
    assert 5 < d < 15 and lo > 0 and stats.verdict(lo, hi) == "WIN"
    assert stats.flips(a, a) == 0
