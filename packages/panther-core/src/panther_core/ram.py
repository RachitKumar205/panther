"""The RAM model: weight bytes + KV cache + runtime overhead.

Pure: it takes an explicit :class:`RuntimeConfig` and never reads global settings.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from panther_core.geometry import ModelGeometry
from panther_core.quant import QuantType

GIB = 1024**3

KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "Q8_0": 34 / 32,
    "F16": 2.0,
}
"""Bytes per KV-cache element for each supported cache type."""


class RuntimeConfig(BaseModel):
    """Everything about RAM use that is not the weight matrices."""

    model_config = ConfigDict(frozen=True)

    context_tokens: int = Field(gt=0)  # KV-cache context length
    kv_cache_type: str  # Q8_0 or F16
    overhead_gib: float = Field(ge=0)  # fixed llama.cpp runtime overhead

    @field_validator("kv_cache_type")
    @classmethod
    def _check_kv(cls, value: str) -> str:
        if value not in KV_BYTES_PER_ELEMENT:
            raise ValueError(
                f"unknown KV cache type {value!r}; choose one of {sorted(KV_BYTES_PER_ELEMENT)}"
            )
        return value


class RamModel(BaseModel):
    """Predicted peak RAM of a model under an assignment of one type per matrix."""

    model_config = ConfigDict(frozen=True)

    geometry: ModelGeometry  # matrices and fixed bytes
    runtime: RuntimeConfig  # context, KV type, overhead

    def kv_bytes(self) -> int:
        return int(
            2
            * self.geometry.spec.num_hidden_layers
            * self.geometry.spec.kv_width
            * self.runtime.context_tokens
            * KV_BYTES_PER_ELEMENT[self.runtime.kv_cache_type]
        )

    def runtime_bytes(self) -> int:
        return self.kv_bytes() + int(self.runtime.overhead_gib * GIB)

    def matrix_bytes(self, assignment: Mapping[str, QuantType]) -> int:
        total = 0
        for matrix in self.geometry.matrices:
            if matrix.name not in assignment:
                raise KeyError(f"no type assigned to {matrix.name}")
            total += matrix.nbytes(assignment[matrix.name])
        return total

    def total_bytes(self, assignment: Mapping[str, QuantType]) -> int:
        return self.geometry.fixed_bytes + self.matrix_bytes(assignment) + self.runtime_bytes()
