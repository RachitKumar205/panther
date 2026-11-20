"""llama-perplexity output parsers, including per-chunk KL (bug B4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from panther_measure.parse import ParseError, parse_choice, parse_kl, parse_ppl, parse_winogrande

FIXTURES = Path(__file__).resolve().parents[2] / "tests/parity/fixtures/llama"


def test_parse_kl_extracts_mean_and_all_48_chunks() -> None:
    metrics = parse_kl((FIXTURES / "kl_q6k_48chunks.log").read_text())
    assert metrics.mean == pytest.approx(0.005810, abs=1e-6)
    assert metrics.top1_pct == pytest.approx(95.453, abs=1e-3)
    assert len(metrics.chunks) == 48
    assert metrics.chunks[0] == pytest.approx(0.00634, abs=1e-5)


def test_parse_ppl_reads_the_final_estimate() -> None:
    assert parse_ppl((FIXTURES / "ppl_type1_q8_5chunks.log").read_text()) == pytest.approx(10.2189, abs=1e-4)


def test_parse_choice_reconstructs_per_question_correctness() -> None:
    text = "\n".join(f"{i}\t{100 * c / i:.8f}" for i, c in enumerate([1, 1, 2, 3], 1))
    assert parse_choice(text, 4).tolist() == [1, 0, 1, 1]


def test_parse_choice_rejects_a_short_file() -> None:
    with pytest.raises(ParseError, match="of 5 questions"):
        parse_choice("1\t50.00000000\n", 5)


def test_parse_winogrande_marks_matching_indices() -> None:
    text = "\n".join(f"{i}\t0.5\t foo bar {a} {b}" for i, (a, b) in enumerate([(1, 1), (1, 2)], 1))
    assert parse_winogrande(text, 2).tolist() == [1, 0]


def test_parse_winogrande_rejects_a_short_file() -> None:
    with pytest.raises(ParseError, match="of 2 questions"):
        parse_winogrande("1\t0.5\t foo bar 1 1\n", 2)
