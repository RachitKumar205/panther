"""The GGUF tensor-name mapping covers every Qwen2 tensor."""

from __future__ import annotations

from panther_quant.load import param_path


def test_top_level_names_map_to_the_model() -> None:
    assert param_path("token_embd.weight") == "embed_tokens.weight"
    assert param_path("output_norm.weight") == "norm.weight"
    assert param_path("output.weight") == "lm_head.weight"


def test_layer_roles_map_to_their_modules() -> None:
    assert param_path("blk.0.attn_norm.weight") == "layers.0.input_layernorm.weight"
    assert param_path("blk.3.attn_q.weight") == "layers.3.self_attn.q_proj.weight"
    assert param_path("blk.3.attn_q.bias") == "layers.3.self_attn.q_proj.bias"
    assert param_path("blk.3.attn_output.weight") == "layers.3.self_attn.o_proj.weight"
    assert param_path("blk.3.ffn_gate.weight") == "layers.3.mlp.gate_proj.weight"
    assert param_path("blk.3.ffn_down.weight") == "layers.3.mlp.down_proj.weight"


def test_unknown_names_return_none() -> None:
    assert param_path("blk.0.attn_q.extra") is None
    assert param_path("something.else.weight") is None
