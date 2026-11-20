"""Per-layer input covariances ``H = E[x^T x]``, captured from the forward pass (GPTQ calibration).

Four covariances per layer: the q/k/v input (`attn_in`), the output-projection input (`o_in`), the
gate/up input (`mlp_in`) and the down-projection input (`down_in`). Computed in fp32 (ADR 0020).
"""

from __future__ import annotations

import torch

from panther_quant.nn import Qwen2Model

# The projection whose input is captured, per covariance key.
_TAPS = {
    "attn_in": "q_proj",
    "o_in": "o_proj",
    "mlp_in": "gate_proj",
    "down_in": "down_proj",
}


def compute_hessians(model: Qwen2Model, ids: torch.Tensor, batch_size: int = 8) -> dict[tuple[int, str], torch.Tensor]:
    """Return ``{(layer, key): H}`` where ``H = mean over tokens of x^T x``, in fp32.

    Args:
        model: The bf16 model (hooks capture its activations).
        ids: ``(n_seq, seq_len)`` token ids; processed in batches of ``batch_size``.
        batch_size: Sequences per forward call.
    """
    hessians: dict[tuple[int, str], torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def capture(key: tuple[int, str]):
        def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            hessians[key] = hessians.get(key, 0.0) + (x.T @ x) / x.shape[0]

        return hook

    for layer_idx, layer in enumerate(model.layers):
        for name, attr in _TAPS.items():
            module = getattr(layer.self_attn if name in ("attn_in", "o_in") else layer.mlp, attr)
            hooks.append(module.register_forward_hook(capture((layer_idx, name))))

    batches = 0
    with torch.no_grad():
        for b in range(0, ids.shape[0], batch_size):
            model(ids[b : b + batch_size])
            batches += 1
    for h in hooks:
        h.remove()
    return {key: value / batches for key, value in hessians.items()}
