"""Panther's pure domain model: geometry, RAM accounting, genomes, scoring and statistics."""

from __future__ import annotations

from panther_core.cache import key
from panther_core.error import weighted_sqerr
from panther_core.geometry import GeometrySpec, Matrix, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType
from panther_core.ram import GIB, KV_BYTES_PER_ELEMENT, RamModel, RuntimeConfig
from panther_core.stats import (
    Score,
    Verdict,
    accuracy,
    flips,
    kl_diff_std_error,
    kl_mean,
    kl_std_error,
    paired_flip_std_error,
    relative_interval,
    relative_score,
    verdict,
)

__all__ = [
    "GIB",
    "IMATRIX_FORMATS",
    "KV_BYTES_PER_ELEMENT",
    "SEARCH_FORMATS",
    "GeometrySpec",
    "Matrix",
    "ModelGeometry",
    "QuantType",
    "RamModel",
    "RuntimeConfig",
    "Score",
    "Verdict",
    "accuracy",
    "flips",
    "key",
    "kl_diff_std_error",
    "kl_mean",
    "kl_std_error",
    "paired_flip_std_error",
    "relative_interval",
    "relative_score",
    "verdict",
    "weighted_sqerr",
]
