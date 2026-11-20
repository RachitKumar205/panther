"""Content-addressed cache keys.

A result is only reusable if it was produced from *every* input that can change it. This module
provides a deterministic hash over arbitrary, ordered-or-not inputs so cache keys are complete and
stable (fixes bug B1: the old keys silently omitted inputs).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import cast

_HEX_DIGITS = 16
"""Length of the returned key (64 bits): plenty for the number of records a run produces."""


def _canonical(value: object) -> object:
    """Normalize ``value`` so equal inputs hash equally regardless of container ordering."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return {"$enum": [type(value).__name__, value.name]}
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value).items()
        return {str(k): _canonical(v) for k, v in sorted(items, key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in cast(Iterable[object], value)]
    if isinstance(value, (set, frozenset)):
        return [_canonical(item) for item in sorted(cast(Iterable[object], value), key=repr)]
    raise TypeError(f"cannot put {type(value).__name__} in a cache key")


def key(*parts: object) -> str:
    """A stable, short content hash of ``parts``.

    Dictionaries and sets are order-insensitive; :class:`~pathlib.Path` and :class:`str` compare
    equal; enums are identified by name; dataclasses by their fields. Callers are responsible for
    including *every* input that affects the result.
    """
    payload = json.dumps(
        [_canonical(part) for part in parts],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HEX_DIGITS]
