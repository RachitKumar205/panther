"""Parsers against real llama-perplexity output captured on the server at the pinned commit."""

import math
from pathlib import Path

import numpy as np
import pytest

from panther.evaluate import (
    ParseError,
    parse_kl,
    parse_ppl_type1,
    parse_type_counts,
    scored_tokens_per_chunk,
)
from panther.stats import Evaluation

FIX = Path(__file__).parent / "fixtures" / "llama"


def test_scored_tokens():
    assert scored_tokens_per_chunk(2048) == 1023


def test_parse_kl_real_output():
    r = parse_kl((FIX / "kl_q6k_48chunks.log").read_text())
    assert r.chunks == 48 and r.n_ctx == 2048
    assert r.kl_mean == pytest.approx(0.005810)
    assert 0 < r.kl_mean_se < r.kl_mean
    assert r.kl_mean < r.kl_p99 < r.kl_max
    assert r.top1_pct == pytest.approx(95.453)
    assert r.ppl_ratio > 1.0


def test_parse_kl_rejects_failed_run():
    # A failed per-chunk reference run: llama.cpp printed an error and exited 0.
    with pytest.raises(ParseError):
        parse_kl((FIX / "kl_base_ref_1chunk.log").read_text())


def test_parse_ppl_type1_real_output():
    r = parse_ppl_type1((FIX / "ppl_type1_q8_5chunks.log").read_text())
    assert r.ppl == pytest.approx(10.2189)
    assert len(r.chunk_mean_nll) == 5 and r.tokens_per_chunk == 1023
    assert r.chunk_mean_nll[0] == pytest.approx(2.638183)
    assert math.exp(r.chunk_mean_nll.mean()) == pytest.approx(10.2189, rel=1e-4)
    assert r.recovery_rel_error < 1e-4


def test_ppl_recovery_is_exact_on_synthetic_running_means():
    rng = np.random.default_rng(0)
    true = rng.normal(2.3, 0.3, 219)
    running = np.cumsum(true) / np.arange(1, 220)
    rows = "\n".join(f"{i * 2048:8d}  {math.exp(m):.4f}  {m:.6f}  0.010000" for i, m in enumerate(running))
    text = (f"perplexity: calculating perplexity over 219 chunks, n_ctx=2048, batch_size=2048\n{rows}\n"
            f"Final estimate: PPL = {math.exp(true.mean()):.4f} +/- 0.1\n")
    r = parse_ppl_type1(text)
    assert np.max(np.abs(r.chunk_mean_nll - true)) < 2.2e-4  # D32 bound at 219 chunks


def test_ppl_type1_rejects_default_output():
    text = ("perplexity: calculating perplexity over 2 chunks, n_ctx=2048, batch_size=2048\n"
            "[1]7.8888,[2]9.0082,\nFinal estimate: PPL = 9.0082 +/- 0.2\n")
    with pytest.raises(ParseError, match="ppl-output-type"):
        parse_ppl_type1(text)


def test_ppl_result_feeds_decision_rule():
    r = parse_ppl_type1((FIX / "ppl_type1_q8_5chunks.log").read_text())
    ev = Evaluation("x", {"t": np.ones(3, dtype=np.int8)}, r.chunk_nll, r.chunk_tokens, 1, 1.0)
    assert ev.perplexity() == pytest.approx(10.2189, rel=1e-4)


def test_type_counts():
    text = "llama_model_loader: - type  f32:  141 tensors\nllama_model_loader: - type q2_K:   18 tensors\n"
    assert parse_type_counts(text) == {"f32": 141, "q2_k": 18}
