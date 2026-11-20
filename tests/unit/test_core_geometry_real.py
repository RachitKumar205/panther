"""Geometry and RAM verified against the vendored model configs."""

from __future__ import annotations

from pathlib import Path

from panther_core.geometry import ModelGeometry
from panther_core.quant import QuantType
from panther_core.ram import RamModel, RuntimeConfig
from panther_data.config import load_model_config

ROOT = Path(__file__).resolve().parents[2]


def geometry(name: str) -> ModelGeometry:
    return ModelGeometry.from_spec(load_model_config(ROOT / "configs/model_configs", name).to_spec())


def test_matrix_count_and_shape_for_all_vendored_models() -> None:
    cases = {
        "Qwen2.5-1.5B-Instruct": (28, 1536, True),
        "Qwen2.5-7B-Instruct": (28, 3584, False),
        "Qwen2.5-14B-Instruct": (48, 5120, False),
        "Mistral-22B-Instruct": (56, 6144, False),
        "Mistral-24B-Instruct": (40, 5120, False),
    }
    for name, (layers, hidden, tied) in cases.items():
        model = geometry(name)
        assert model.spec.num_hidden_layers == layers
        assert model.spec.hidden_size == hidden
        assert len(model.matrices) == (1 if tied else 2) + 7 * layers


def test_mistral_has_explicit_head_dim_and_no_qkv_bias() -> None:
    model = geometry("Mistral-22B-Instruct")
    assert model.spec.head_dim == 128
    hidden = model.spec.hidden_size
    assert model.fixed_bytes == 4 * (56 * (2 * hidden) + hidden)  # bias = 0


def test_qwen2_has_qkv_biases() -> None:
    model = geometry("Qwen2.5-7B-Instruct")
    hidden = model.spec.hidden_size
    bias = model.spec.attention_width + 2 * model.spec.kv_width
    assert model.fixed_bytes == 4 * (28 * (2 * hidden + bias) + hidden)


def test_qwen_7b_ram_is_in_a_plausible_range() -> None:
    model = geometry("Qwen2.5-7B-Instruct")
    ram = RamModel(
        geometry=model,
        runtime=RuntimeConfig(context_tokens=4096, kv_cache_type="Q8_0", overhead_gib=0.4),
    )
    assignment = {m.name: QuantType.Q4_K for m in model.matrices}
    total = ram.total_bytes(assignment)
    assert 4 * 2**30 < total < 6 * 2**30
