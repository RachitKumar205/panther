"""llama.cpp output parsers (real output captured on the server at the pinned commit) and GSM8K scoring."""

import math
from pathlib import Path

import numpy as np
import pytest

from panther.evaluate import (ParseError, gsm8k_correct, gsm8k_prompt, parse_kl, parse_multiple_choice, parse_ppl,
                              parse_winogrande)
from panther.stats import Evaluation

FIX = Path(__file__).parent / "fixtures" / "llama"


def test_kl_real_output():
    r = parse_kl((FIX / "kl_q6k_48chunks.log").read_text())
    assert r["kl_mean"] == pytest.approx(0.005810) and r["top1_pct"] == pytest.approx(95.453)
    assert r["kl_mean"] < r["kl_p99"] and r["ppl_ratio"] > 1.0


def test_kl_rejects_a_failed_run():  # llama.cpp printed an error and exited 0
    with pytest.raises(ParseError):
        parse_kl((FIX / "kl_base_ref_1chunk.log").read_text())


def test_ppl_real_output_and_per_chunk_recovery():
    ppl, chunk, tokens = parse_ppl((FIX / "ppl_type1_q8_5chunks.log").read_text())
    assert ppl == pytest.approx(10.2189) and len(chunk) == 5 and tokens == 1023
    assert math.exp(chunk.mean()) == pytest.approx(10.2189, rel=1e-4)
    ev = Evaluation("x", {"t": np.ones(3, dtype=np.int8)}, chunk * tokens, np.full(5, tokens), 1, 1.0)
    assert ev.perplexity() == pytest.approx(10.2189, rel=1e-4)


def test_ppl_rejects_output_without_per_chunk_rows():
    text = ("perplexity: calculating perplexity over 2 chunks, n_ctx=2048, batch_size=2048\n"
            "[1]7.8888,[2]9.0082,\nFinal estimate: PPL = 9.0082 +/- 0.2\n")
    with pytest.raises(ParseError):
        parse_ppl(text)


def _mc_output(correct: list[int]) -> str:
    rows, c = [], 0
    for i, x in enumerate(correct, 1):
        c += x
        rows.append(f"{i}\t{100 * c / i:.8f}")
    return "\n".join(rows) + f"\n\n Final result: {100 * c / len(correct):.4f} +/- 0.1\n"


def test_multiple_choice_recovers_every_answer_and_catches_dropped_items():
    correct = np.random.default_rng(1).integers(0, 2, 14042).tolist()
    assert parse_multiple_choice(_mc_output(correct), len(correct)).tolist() == correct
    with pytest.raises(ParseError):
        parse_multiple_choice(_mc_output(correct[:-1]), len(correct))


def test_winogrande_rows():
    text = ("1\t100.0000\t -1.000000   -2.000000  1  1\n2\t50.0000\t -1.500000   -0.500000  2  1\n\n"
            "Final Winogrande score(2 tasks): 50.0000 +/- 50.0000\n")
    assert parse_winogrande(text, 2).tolist() == [1, 0]
    with pytest.raises(ParseError):
        parse_winogrande(text, 3)


def test_gsm8k_strict_match():
    assert gsm8k_correct("so 3 + 4 = 7\n#### 7", "Natalia sold 7 clips.\n#### 7")
    assert gsm8k_correct("#### 1,000.", "#### 1000")
    assert not gsm8k_correct("the answer is 7", "#### 7")  # no "####": wrong under strict match
    assert gsm8k_prompt("q?", [{"question": "a?", "answer": "b"}]) == "Question: a?\nAnswer: b\n\nQuestion: q?\nAnswer:"
