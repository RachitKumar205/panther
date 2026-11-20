"""The verdict is the relative score's interval against the preset."""

from __future__ import annotations

import pytest

from panther_pipeline.verdict import Measurement, score, versus


def make(correct: list[bool], kl_chunks: tuple[float, ...]) -> Measurement:
    return Measurement(correct={"mmlu": correct}, kl_chunks=kl_chunks)


def test_verdict_wins_when_clearly_better() -> None:
    original = {"mmlu": [True] * 100}
    ours = make([True] * 100, (0.1, 0.1))
    preset = make([True] * 50 + [False] * 50, (0.4, 0.4))
    result = versus(ours, preset, original)
    assert result["verdict"] == "WIN"
    assert result["relative"] < 100


def test_verdict_ties_when_the_interval_crosses_100() -> None:
    original = {"mmlu": [True] * 100}
    ours = make([True] * 100, (0.1, 0.3))  # KL mean 0.2 with a nonzero chunk SE
    preset = make([True] * 100, (0.2, 0.2))  # same KL mean, zero SE
    result = versus(ours, preset, original)
    assert result["verdict"] == "TIE"
    assert result["ci"][0] < 100 < result["ci"][1]


def test_score_is_flips_plus_weighted_kl() -> None:
    original = {"mmlu": [True] * 100}
    m = make([True] * 75 + [False] * 25, (0.3, 0.3))  # 25% flips, KL 0.3
    assert score(m, original, kl_weight=10.0) == pytest.approx(25.0 + 10 * 0.3)
