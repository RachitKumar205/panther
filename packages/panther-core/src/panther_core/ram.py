"""The RAM model: weight bytes + KV cache + runtime overhead.

This is the model the search window, the claim and the dashboard must agree on. It is pure: it
takes an explicit :class:`RuntimeConfig` and never reads global settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from panther_core.geometry import ModelGeometry
from panther_core.quant import QuantType

GIB = 1024**3

KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "Q8_0": 34 / 32,
    "F16": 2.0,
}
"""Bytes per KV-cache element for each supported cache type."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Everything about RAM use that is not the weight matrices."""

    context_tokens: int
    kv_cache_type: str
    overhead_gib: float

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError(f"context_tokens must be positive, got {self.context_tokens}")
        if self.kv_cache_type not in KV_BYTES_PER_ELEMENT:
            raise ValueError(
                f"unknown KV cache type {self.kv_cache_type!r}; "
                f"choose one of {sorted(KV_BYTES_PER_ELEMENT)}"
            )
        if self.overhead_gib < 0:
            raise ValueError(f"overhead_gib must be non-negative, got {self.overhead_gib}")


@dataclass(frozen=True, slots=True)
class RamModel:
    """Predicted peak RAM of a model under an assignment of one type per matrix."""

    geometry: ModelGeometry
    runtime: RuntimeConfig

    def kv_bytes(self) -> int:
        """Bytes of the KV cache at the configured context."""
        return int(
            2
            * self.geometry.spec.num_hidden_layers
            * self.geometry.spec.kv_width
            * self.runtime.context_tokens
            * KV_BYTES_PER_ELEMENT[self.runtime.kv_cache_type]
        )

    def runtime_bytes(self) -> int:
        """Bytes besides the weights: KV cache and fixed overhead."""
        return self.kv_bytes() + int(self.runtime.overhead_gib * GIB)

    def matrix_bytes(self, assignment: Mapping[str, QuantType]) -> int:
        """Bytes of every matrix under ``assignment``.

        Raises:
            KeyError: If a matrix has no assigned type.
        """
        total = 0
        for matrix in self.geometry.matrices:
            if matrix.name not in assignment:
                raise KeyError(f"no type assigned to {matrix.name}")
            total += matrix.nbytes(assignment[matrix.name])
        return total

    def total_bytes(self, assignment: Mapping[str, QuantType]) -> int:
        """Predicted peak RAM in bytes: fixed bytes + matrices + runtime."""
        return self.geometry.fixed_bytes + self.matrix_bytes(assignment) + self.runtime_bytes()
