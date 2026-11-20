"""Config files validate and map onto the core domain types."""

from __future__ import annotations

from pathlib import Path

from panther_core.geometry import ModelGeometry
from panther_core.quant import QuantType
from panther_data.config import load_model_config, load_models, load_search_config

ROOT = Path(__file__).resolve().parents[2]


def test_qwen_geometry_maps_to_the_expected_matrices() -> None:
    spec = load_model_config(ROOT / "configs/model_configs", "Qwen2.5-7B-Instruct").to_spec()
    assert spec.num_hidden_layers == 28
    assert spec.hidden_size == 3584
    assert len(ModelGeometry.from_spec(spec).matrices) == 2 + 7 * 28  # untied: embedding + head


def test_search_config_maps_to_runtime_and_formats() -> None:
    config = load_search_config(ROOT / "configs/search.yaml")
    assert config.runtime.context_tokens == 4096
    assert len(config.formats) == 14
    assert config.requires_imatrix == [QuantType.IQ1_S, QuantType.IQ1_M, QuantType.IQ2_XXS, QuantType.IQ2_XS]


def test_models_config_lists_every_model_and_the_llama_build() -> None:
    models = load_models(ROOT / "configs/models.yaml")
    assert len(models.models) == 5
    assert models.llama_cpp.commit
