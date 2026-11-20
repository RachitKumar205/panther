"""The cache reuses a result only when every input matches (bug B1)."""

from __future__ import annotations

from panther_measure.cache import ResultCache


def test_cache_reuses_on_identical_inputs() -> None:
    cache = ResultCache()
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        return 42

    assert cache.get_or_run("m", "hash", "commit", compute=compute) == (42, False)
    assert cache.get_or_run("m", "hash", "commit", compute=compute) == (42, True)
    assert calls == 1


def test_cache_recomputes_when_any_input_changes() -> None:
    cache = ResultCache()
    assert cache.get_or_run("m", "hash", "commit", compute=lambda: 1) == (1, False)
    assert cache.get_or_run("m", "hash", "other", compute=lambda: 2) == (2, False)


def test_cache_is_order_insensitive_for_mappings() -> None:
    cache = ResultCache()
    assert cache.get_or_run({"a": 1, "b": 2}, compute=lambda: 1) == (1, False)
    assert cache.get_or_run({"b": 2, "a": 1}, compute=lambda: 1) == (1, True)
