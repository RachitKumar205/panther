"""Load a 16-bit GGUF into :class:`~panther_quant.nn.Qwen2Model`.

The name mapping understands Qwen2's GGUF tensor naming (``convert_hf_to_gguf`` layout); the bytes
are dequantized with gguf (ADR 0018) and transposed from ``(in, out)`` to torch's ``(out, in)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gguf.quants import dequantize
from numpy.typing import NDArray

from panther_quant.nn import Qwen2Model

# role -> (module path within a layer, allowed suffix kinds)
_ROLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "attn_norm": ("input_layernorm", ("weight",)),
    "attn_q": ("self_attn.q_proj", ("weight", "bias")),
    "attn_k": ("self_attn.k_proj", ("weight", "bias")),
    "attn_v": ("self_attn.v_proj", ("weight", "bias")),
    "attn_output": ("self_attn.o_proj", ("weight",)),
    "ffn_norm": ("post_attention_layernorm", ("weight",)),
    "ffn_gate": ("mlp.gate_proj", ("weight",)),
    "ffn_up": ("mlp.up_proj", ("weight",)),
    "ffn_down": ("mlp.down_proj", ("weight",)),
}

_TOP_LEVEL = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
    "output.weight": "lm_head.weight",
}


def param_path(name: str) -> str | None:
    """Map a Qwen2 GGUF tensor name to a ``Qwen2Model`` parameter path, or ``None``."""
    if name in _TOP_LEVEL:
        return _TOP_LEVEL[name]
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[2] in _ROLE:
        layer, role, kind = parts[1], parts[2], parts[3]
        if kind in _ROLE[role][1]:
            return f"layers.{layer}.{_ROLE[role][0]}.{kind}"
    return None


def load_qwen2(path: Path, model: Qwen2Model) -> Qwen2Model:
    """Load the GGUF's weights into ``model`` and return it (``load_state_dict``, strict)."""
    import gguf

    reader = gguf.GGUFReader(path)
    state: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        target = param_path(tensor.name)
        if target is None:
            continue
        data: NDArray[np.float32] = dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32)
        if tensor.name.endswith(".weight") and data.ndim == 2:
            data = data.T  # GGUF stores (in, out); nn.Linear wants (out, in)
        state[target] = torch.tensor(data)
    model.load_state_dict(state, strict=True)
    return model
