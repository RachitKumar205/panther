"""A tiny transformers model saved as the download (hf/) and the same weights as the 16-bit original GGUF, the way
llama.cpp's convert_hf_to_gguf.py writes them (q/k rows reordered for llama-style models)."""

from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from tests.model.tiny import HEADS, HIDDEN, INTER, KV_HEADS, LAYERS, TOKENS, VOCAB

ROLES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
         "mlp.down_proj": "ffn_down", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}


def llama_cpp_permute(weights: np.ndarray, n_head: int, n_head_kv: int | None) -> np.ndarray:
    """convert_hf_to_gguf's LlamaModel.permute at the pinned llama.cpp commit, copied as it is."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def make(hf: Path, original: Path, llama: bool = False) -> torch.nn.Module:
    """Saves the model to hf and its weights to original; returns it. Every weight is bf16-exact, so the 16-bit file
    holds exactly the model's values."""
    torch.manual_seed(0)
    shape = {"hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
             "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB,
             "max_position_embeddings": 4096, "tie_word_embeddings": False}
    net = LlamaForCausalLM(LlamaConfig(**shape)) if llama else Qwen2ForCausalLM(Qwen2Config(**shape))
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "norm" in name:
                p.copy_(torch.empty_like(p).uniform_(0.5, 1.5))
            p.copy_(p.to(torch.bfloat16).float())
    net.save_pretrained(hf)

    arch = "llama" if llama else "qwen2"
    w = gguf.GGUFWriter(original, arch)
    w.add_context_length(4096)
    w.add_embedding_length(HIDDEN)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1e4)
    w.add_rope_dimension_count(HIDDEN // HEADS)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(TOKENS)
    w.add_token_types([1] * VOCAB)
    w.add_token_merges(["t h"])

    def put(name: str, value: torch.Tensor) -> None:
        a = value.detach().float().numpy()
        if a.ndim == 2:  # matrices in bf16 (exact: the values are bf16 already)
            w.add_tensor(name, (a.view(np.uint32) >> 16).astype(np.uint16), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, a)

    put("token_embd.weight", net.model.embed_tokens.weight)
    for i, layer in enumerate(net.model.layers):
        for module, role in ROLES.items():
            value = layer.get_submodule(module).weight
            if llama and role in ("attn_q", "attn_k"):
                value = torch.from_numpy(llama_cpp_permute(value.detach().numpy(), HEADS,
                                                           HEADS if role == "attn_q" else KV_HEADS))
            put(f"blk.{i}.{role}.weight", value)
        if not llama:
            for module, role in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v")):
                put(f"blk.{i}.{role}.bias", layer.self_attn.get_submodule(module).bias)
    put("output_norm.weight", net.model.norm.weight)
    put("output.weight", net.lm_head.weight)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return net
