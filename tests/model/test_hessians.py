"""H: the mean of x xᵀ over every calibration token, for every input of every layer."""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from panther.model.gptq import collect_hessians, input_of


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=64)
    return Qwen2ForCausalLM(c).eval()


def ids() -> torch.Tensor:
    return torch.randint(0, 100, (5, 16), generator=torch.Generator().manual_seed(1))


def test_h_is_the_mean_outer_product_of_what_each_input_sees() -> None:
    net, x = tiny_model(), ids()
    seen = {}
    mod = net.model.layers[1].mlp.down_proj
    hook = mod.register_forward_pre_hook(lambda _, i: seen.setdefault("x", []).append(i[0].reshape(-1, 128)))
    with torch.no_grad():
        net(x)
    hook.remove()
    inputs = torch.cat(seen["x"])
    h = collect_hessians(net, x, ["model.layers.1.mlp.down_proj"], batch=2)["model.layers.1.mlp.down_proj"]
    assert torch.allclose(h, inputs.T @ inputs / len(inputs), atol=1e-5)


def test_the_output_layer_reads_the_final_hidden_state() -> None:
    net, x = tiny_model(), ids()
    seen = []
    hook = net.lm_head.register_forward_pre_hook(lambda _, i: seen.append(i[0].reshape(-1, 64)))
    with torch.no_grad():
        net(x)
    hook.remove()
    inputs = torch.cat(seen)
    h = collect_hessians(net, x, ["lm_head"])["lm_head"]
    assert torch.allclose(h, inputs.T @ inputs / len(inputs), atol=1e-5)


def test_each_gguf_matrix_maps_to_the_module_whose_input_it_reads() -> None:
    assert input_of("blk.3.attn_k.weight") == "model.layers.3.self_attn.q_proj"
    assert input_of("blk.0.ffn_up.weight") == "model.layers.0.mlp.gate_proj"
    assert input_of("blk.12.ffn_down.weight") == "model.layers.12.mlp.down_proj"
    assert input_of("output.weight") == "lm_head"
    for name in ["model.layers.2.self_attn.q_proj", "model.layers.2.self_attn.o_proj", "model.layers.2.mlp.gate_proj",
                 "model.layers.2.mlp.down_proj", "lm_head"]:
        tiny_model().get_submodule(name)  # every module input_of names exists in a transformers model


def test_h_is_symmetric_and_grouping_does_not_change_it() -> None:
    net, x = tiny_model(), ids()
    names = [f"model.layers.{i}.{m}" for i in range(3) for m in ("self_attn.q_proj", "mlp.down_proj")] + ["lm_head"]
    together = collect_hessians(net, x, names)
    apart = {k: v for name in names for k, v in collect_hessians(net, x, [name]).items()}
    assert together.keys() == apart.keys() == set(names)
    for name, h in together.items():
        assert torch.allclose(h, h.T, atol=1e-6) and torch.allclose(h, apart[name])


def test_hooks_are_removed_afterwards() -> None:
    net = tiny_model()
    collect_hessians(net, ids(), ["model.layers.0.self_attn.q_proj"])
    assert not net.model.layers[0].self_attn.q_proj._forward_pre_hooks
