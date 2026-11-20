"""The RAM model: weights + KV cache + overhead."""

from __future__ import annotations

import pytest

from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import QuantType
from panther_core.ram import GIB, KV_BYTES_PER_ELEMENT, RamModel, RuntimeConfig


def spec() -> GeometrySpec:
    # Every input width (hidden_size, attention, intermediate) is a multiple of the largest
    # block size (256), so the block-based types are all representable.
    return GeometrySpec(
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        vocab_size=320,
        head_dim=32,
        tie_word_embeddings=True,
        model_type="qwen2",
    )


def build() -> tuple[ModelGeometry, RuntimeConfig, RamModel]:
    geometry = ModelGeometry.from_spec(spec())
    runtime = RuntimeConfig(context_tokens=4096, kv_cache_type="Q8_0", overhead_gib=0.4)
    return geometry, runtime, RamModel(geometry=geometry, runtime=runtime)


def uniform(model: ModelGeometry, quant: QuantType) -> dict[str, QuantType]:
    return {matrix.name: quant for matrix in model.matrices}


def test_kv_bytes_follow_layers_kv_width_context_and_element_size() -> None:
    geometry, runtime, model = build()
    expected = int(
        2
        * geometry.spec.num_hidden_layers
        * geometry.spec.kv_width
        * runtime.context_tokens
        * KV_BYTES_PER_ELEMENT["Q8_0"]
    )
    assert model.kv_bytes() == expected


def test_runtime_bytes_adds_overhead() -> None:
    _, runtime, model = build()
    assert model.runtime_bytes() == model.kv_bytes() + int(runtime.overhead_gib * GIB)


def test_total_is_fixed_plus_matrices_plus_runtime() -> None:
    geometry, _, model = build()
    assignment = uniform(geometry, QuantType.Q4_K)
    matrix_bytes = sum(matrix.nbytes(QuantType.Q4_K) for matrix in geometry.matrices)
    assert model.matrix_bytes(assignment) == matrix_bytes
    assert model.total_bytes(assignment) == geometry.fixed_bytes + matrix_bytes + model.runtime_bytes()


def test_a_higher_type_never_uses_less_memory() -> None:
    geometry, _, model = build()
    low = model.total_bytes(uniform(geometry, QuantType.Q2_K))
    high = model.total_bytes(uniform(geometry, QuantType.Q8_0))
    assert low < high


def test_missing_assignment_raises() -> None:
    geometry, _, model = build()
    assignment = uniform(geometry, QuantType.Q4_K)
    del assignment[geometry.matrices[0].name]
    with pytest.raises(KeyError, match="no type assigned"):
        model.matrix_bytes(assignment)


def test_unknown_kv_cache_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown KV cache type"):
        RuntimeConfig(context_tokens=4096, kv_cache_type="Q4_0", overhead_gib=0.4)


def test_invalid_runtime_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        RuntimeConfig(context_tokens=0, kv_cache_type="Q8_0", overhead_gib=0.4)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        RuntimeConfig(context_tokens=4096, kv_cache_type="Q8_0", overhead_gib=-1)


def test_f16_kv_cache_is_larger_than_q8() -> None:
    geometry, _, _ = build()
    q8 = RamModel(geometry=geometry, runtime=RuntimeConfig(context_tokens=4096, kv_cache_type="Q8_0", overhead_gib=0))
    f16 = RamModel(geometry=geometry, runtime=RuntimeConfig(context_tokens=4096, kv_cache_type="F16", overhead_gib=0))
    assert f16.kv_bytes() > q8.kv_bytes()
