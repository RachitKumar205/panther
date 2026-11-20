"""Content-addressed cache keys are deterministic and order-insensitive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from panther_core.cache import key
from panther_core.quant import QuantType


def test_same_inputs_give_the_same_key() -> None:
    assert key("a", 1, 2.5, True, None) == key("a", 1, 2.5, True, None)


def test_dictionary_order_does_not_matter() -> None:
    assert key({"a": 1, "b": 2}) == key({"b": 2, "a": 1})


def test_set_order_does_not_matter() -> None:
    assert key({1, 2, 3}) == key({3, 2, 1})


def test_list_order_does_matter() -> None:
    assert key([1, 2]) != key([2, 1])


def test_path_and_string_are_equal() -> None:
    assert key(Path("a/b")) == key("a/b")


def test_enum_is_identified_by_name() -> None:
    assert key(QuantType.Q4_K) == key({"$enum": ("QuantType", "Q4_K")})


def test_bytes_are_hashed_by_content() -> None:
    assert key(b"abc") == key(b"abc")
    assert key(b"abc") != key(b"abd")


@dataclass(frozen=True, slots=True)
class _Thing:
    name: str
    count: int


def test_dataclass_is_hashed_by_fields() -> None:
    assert key(_Thing("x", 1)) == key({"name": "x", "count": 1})


def test_distinct_inputs_give_distinct_keys() -> None:
    assert len({key(i) for i in range(100)}) == 100


def test_unknown_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="cache key"):
        key(object())
