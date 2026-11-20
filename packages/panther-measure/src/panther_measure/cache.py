"""A result cache keyed by a complete input digest (fixes bug B1).

The old cache reused results when only some inputs matched. Here the caller passes every input that
can change the result and the key hashes all of them, so a stale result can never be reused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from pydantic import BaseModel, Field

from panther_core.cache import key

T = TypeVar("T")


class ResultCache(BaseModel):
    """In-memory cache from a complete input digest to a computed value."""

    entries: dict[str, object] = Field(default_factory=dict)  # key -> cached value

    def get_or_run(self, *inputs: object, compute: Callable[[], T]) -> tuple[T, bool]:
        """Return the cached value for ``inputs``, or compute and cache it.

        Args:
            inputs: Every input that can change the result (file hashes, llama commit, constants).
            compute: Called only on a miss.

        Returns:
            ``(value, hit)`` where ``hit`` is True when the value came from the cache.
        """
        k = key(*inputs)
        if k in self.entries:
            return cast(T, self.entries[k]), True
        value = compute()
        self.entries[k] = value
        return value, False
