"""The guard pauses after 3 bad checks and resumes after 5 good ones."""

from __future__ import annotations

from panther_ops.guard import Guard


def test_guard_pauses_after_three_bad_checks() -> None:
    guard = Guard()
    assert guard.tick(False) is None
    assert guard.tick(False) is None
    assert guard.tick(False) == "pause"
    assert guard.paused


def test_guard_resumes_after_five_good_checks() -> None:
    guard = Guard()
    for _ in range(3):
        guard.tick(False)
    assert guard.tick(True) is None
    assert guard.tick(True) is None
    assert guard.tick(True) is None
    assert guard.tick(True) is None
    assert guard.tick(True) == "resume"
    assert not guard.paused


def test_a_good_check_resets_the_bad_counter() -> None:
    guard = Guard()
    guard.tick(False)  # bad = 1
    guard.tick(False)  # bad = 2
    guard.tick(True)  # resets bad
    assert guard.tick(False) is None  # bad = 1
    assert guard.tick(False) is None  # bad = 2
    assert guard.tick(False) == "pause"  # bad = 3
