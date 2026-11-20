"""Scoring and statistics, including the bug B4 fixes."""

from __future__ import annotations

import numpy as np
import pytest

from panther_core.stats import (
    Score,
    Verdict,
    accuracy,
    flips,
    kl_diff_std_error,
    kl_std_error,
    paired_flip_std_error,
    relative_interval,
    relative_score,
    verdict,
)


def test_accuracy_is_the_mean_task_accuracy_in_percent() -> None:
    correct = {"mmlu": [True, False, True, True], "arc": [True, True]}
    assert accuracy(correct) == pytest.approx(100 * (0.75 + 1.0) / 2)


def test_flips_counts_questions_whose_answer_changed() -> None:
    original = {"mmlu": [True, True, True, True]}
    ours = {"mmlu": [True, False, True, True]}
    assert flips(ours, original) == pytest.approx(25.0)
    assert flips(original, original) == 0.0


def test_flips_ignores_tasks_absent_from_the_original() -> None:
    original = {"mmlu": [True, True]}
    assert flips({"mmlu": [False, False], "extra": [True]}, original) == pytest.approx(100.0)


def test_paired_flip_std_error_is_zero_for_identical_files() -> None:
    original = {"mmlu": [True, False] * 20}
    assert paired_flip_std_error(original, original, original) == pytest.approx(0.0)


def test_paired_flip_std_error_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    original = {"mmlu": rng.integers(0, 2, 1000).astype(bool).tolist()}
    ours = {"mmlu": rng.integers(0, 2, 1000).astype(bool).tolist()}
    theirs = {"mmlu": rng.integers(0, 2, 1000).astype(bool).tolist()}
    assert paired_flip_std_error(ours, theirs, original) == pytest.approx(
        paired_flip_std_error(theirs, ours, original)
    )


def test_kl_std_error_is_the_chunk_mean_standard_error() -> None:
    chunks = [1.0, 2.0, 3.0, 4.0]
    expected = np.std(chunks, ddof=1) / 2
    assert kl_std_error(chunks) == pytest.approx(expected)


def test_kl_diff_std_error_is_paired_over_the_same_chunks() -> None:
    ours = [1.0, 2.0, 3.0]
    theirs = [1.1, 2.1, 3.1]
    assert kl_diff_std_error(ours, theirs) == pytest.approx(kl_std_error([0.1, 0.1, 0.1]))
    # Identical chunks have no difference at all.
    assert kl_diff_std_error(ours, ours) == pytest.approx(0.0)


def test_score_value_and_standard_error() -> None:
    score = Score(flips=5.0, kl=0.2, kl_weight=10.0, flips_se=1.0, kl_se=0.05)
    assert score.value == pytest.approx(5.0 + 10 * 0.2)
    assert score.std_error == pytest.approx(np.hypot(1.0, 10 * 0.05))


def test_relative_score_and_interval() -> None:
    assert relative_score(50.0, 100.0) == pytest.approx(50.0)
    lower, upper = relative_interval(ours=50.0, preset=100.0, std_error=10.0, z=1.96)
    assert lower == pytest.approx(100 * (50 - 19.6) / 100)
    assert upper == pytest.approx(100 * (50 + 19.6) / 100)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (50.0, 90.0, Verdict.WIN),
        (110.0, 130.0, Verdict.LOSE),
        (90.0, 110.0, Verdict.TIE),
    ],
)
def test_verdict_from_the_interval(lower: float, upper: float, expected: Verdict) -> None:
    assert verdict(lower, upper) is expected
