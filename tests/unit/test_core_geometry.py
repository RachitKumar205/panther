"""Model geometry from a typed spec."""

from __future__ import annotations

import pytest

from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS


def spec(**overrides: object) -> GeometrySpec:
    values: dict[str, object] = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 10,
        "head_dim": 4,
        "tie_word_embeddings": True,
        "model_type": "qwen2",
    }
    values.update(overrides)
    return GeometrySpec(**values)  # type: ignore[arg-type]


def geometry(**overrides: object) -> ModelGeometry:
    return ModelGeometry.from_spec(spec(**overrides))


def test_matrix_count_for_a_tied_model() -> None:
    model = geometry()
    assert len(model.matrices) == 1 + 7 * model.spec.num_hidden_layers
    assert [matrix.name for matrix in model.matrices[:2]] == ["token_embd.weight", "blk.0.attn_q.weight"]


def test_shapes_follow_the_attention_geometry() -> None:
    model = geometry()
    by_name = {matrix.name: matrix for matrix in model.matrices}
    assert (by_name["blk.0.attn_q.weight"].rows, by_name["blk.0.attn_q.weight"].cols) == (8, 8)
    assert (by_name["blk.0.attn_k.weight"].rows, by_name["blk.0.attn_k.weight"].cols) == (4, 8)
    assert (by_name["blk.0.attn_output.weight"].rows, by_name["blk.0.attn_output.weight"].cols) == (8, 8)
    assert (by_name["blk.0.ffn_down.weight"].rows, by_name["blk.0.ffn_down.weight"].cols) == (8, 16)


def test_parameter_count() -> None:
    model = geometry()
    expected = sum(matrix.rows * matrix.cols for matrix in model.matrices)
    assert model.params == expected


def test_fixed_bytes_for_qwen2_includes_qkv_biases() -> None:
    model = geometry()
    attention, kv, hidden = 8, 4, 8
    bias = attention + 2 * kv
    expected = 4 * (2 * (2 * hidden + bias) + hidden)
    assert model.fixed_bytes == expected


def test_fixed_bytes_for_a_model_without_biases() -> None:
    model = geometry(model_type="mistral")
    expected = 4 * (2 * (2 * 8) + 8)
    assert model.fixed_bytes == expected


def test_tied_model_has_no_output_matrix() -> None:
    names = {matrix.name for matrix in geometry().matrices}
    assert "output.weight" not in names


def test_untied_model_has_an_output_matrix_and_no_imatrix_for_it() -> None:
    model = geometry(tie_word_embeddings=False)
    names = {matrix.name for matrix in model.matrices}
    assert "output.weight" in names
    assert model.no_imatrix == frozenset({"token_embd.weight", "output.weight"})


def test_imatrix_process_output_lets_the_output_layer_use_imatrix_types() -> None:
    model = geometry(tie_word_embeddings=False, imatrix_process_output=True)
    assert model.no_imatrix == frozenset({"token_embd.weight"})


def test_head_dim_can_be_derived_from_hidden_size() -> None:
    model = geometry(head_dim=None, hidden_size=8, num_attention_heads=2)
    assert model.spec.head_width == 4


def test_imatrix_formats_are_excluded_for_matrices_without_imatrix_data() -> None:
    allowed = geometry(tie_word_embeddings=False).allowed(SEARCH_FORMATS, IMATRIX_FORMATS)
    assert not (set(allowed["token_embd.weight"]) & IMATRIX_FORMATS)
    assert not (set(allowed["output.weight"]) & IMATRIX_FORMATS)
    assert set(allowed["blk.0.attn_q.weight"]) >= IMATRIX_FORMATS


def test_allowed_types_are_sorted_low_to_high() -> None:
    allowed = geometry().allowed(SEARCH_FORMATS, IMATRIX_FORMATS)
    for types in allowed.values():
        bits = [quant.bits_per_weight for quant in types]
        assert bits == sorted(bits)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GeometrySpec(
            num_hidden_layers=1,
            hidden_size=10,
            num_attention_heads=3,
            num_key_value_heads=1,
            intermediate_size=4,
            vocab_size=4,
        )
    with pytest.raises(ValueError, match="greater than 0"):
        GeometrySpec(
            num_hidden_layers=0,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=4,
            vocab_size=4,
        )
