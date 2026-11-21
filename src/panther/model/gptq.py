"""Error-compensated rounding (GPTQ) into llama.cpp's own types.

Plain rounding rounds every weight on its own. GPTQ rounds 256 columns at a time with llama.cpp's quantizer, then
moves the rounding error onto the columns not yet rounded, weighted by how the matrix's inputs correlate (H), so the
matrix's *output* changes as little as possible.

The maths runs in float32. The weights come from the 16-bit original and convert to float32 exactly; float32 is what
llama.cpp's quantizer takes, and a Cholesky factorisation in 16 bits would be too imprecise.
"""

from __future__ import annotations

import numpy as np
import torch

from panther.model.quantizer import Quantizer

BLOCK = 256  # columns rounded together: one block of the K and IQ types
DAMPING = 0.01  # added to H's diagonal, as a share of its mean: keeps the inverse stable

# The input each layer matrix reads, named by the transformers module it enters (the same in Qwen2 and Mistral).
# q/k/v read one input, gate/up another.
LAYER_INPUTS = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.q_proj", "attn_v": "self_attn.q_proj",
                "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.gate_proj",
                "ffn_down": "mlp.down_proj"}


def input_of(matrix: str) -> str:
    """The transformers module whose input a GGUF matrix reads: blk.3.attn_k.weight → model.layers.3.self_attn.q_proj;
    output.weight (the output layer) → lm_head, which reads the final hidden state."""
    if matrix == "output.weight":
        return "lm_head"
    _, layer, role, _ = matrix.split(".")
    return f"model.layers.{layer}.{LAYER_INPUTS[role]}"


class GPTQ:
    """GPTQ for one input: built once from its H, then rounds every matrix reading that input, in every type."""

    def __init__(self, h: torch.Tensor) -> None:
        """h: the mean of x xᵀ over the calibration tokens (columns x columns)."""
        self.h = h.clone().float()
        h = self.h.clone()
        self.importance = torch.diag(h).clone()  # each column's mean squared input; 0: a column never used
        unused = self.importance == 0
        h[unused, unused] = 1  # keeps H invertible; those columns' weights are zeroed in round()
        h += DAMPING * torch.mean(torch.diag(h)) * torch.eye(len(h), device=h.device)
        self.u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)

    def round(self, w: torch.Tensor, type_: str, quantizer: Quantizer) -> tuple[np.ndarray, np.ndarray]:
        """(codes, values) of w (rows x columns) in type_, each block's error moved onto the columns after it. values:
        what the codes decode to."""
        w = w.clone().float()
        w[:, self.importance == 0] = 0
        codes, decoded = [], []
        for a in range(0, w.shape[1], BLOCK):
            b = a + BLOCK
            block_codes, values = quantizer.quantize(w[:, a:b].cpu().numpy(), self.importance[a:b].cpu().numpy(), type_)
            codes.append(block_codes)
            decoded.append(values)
            error = torch.linalg.solve_triangular(self.u[a:b, a:b], w[:, a:b] - torch.from_numpy(values).to(w.device),
                                                  upper=True, left=False)
            w[:, b:] -= error @ self.u[a:b, b:]
        return np.concatenate(codes, axis=1), np.concatenate(decoded, axis=1)

    def output_error(self, w: torch.Tensor, values: np.ndarray) -> float:
        """How much the matrix's output changes, averaged over the calibration tokens: E[‖(W − Q)x‖²] =
        trace((W − Q) H (W − Q)ᵀ). The search's repair steps by it."""
        d = w.float() - torch.from_numpy(values).to(w.device)
        return float(((d @ self.h) * d).sum())


def collect_hessians(model: torch.nn.Module, ids: torch.Tensor, modules: list[str],
                     batch: int = 8) -> dict[str, torch.Tensor]:
    """H for each of the named modules' inputs: the mean of x xᵀ over every token of ids (sequences x tokens), x
    being what the model feeds that module."""
    sums: dict[str, torch.Tensor] = {}
    hooks = []
    for name in modules:
        def add(_module, inputs, name=name):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            sums[name] = sums.get(name, 0) + x.T @ x
        hooks.append(model.get_submodule(name).register_forward_pre_hook(add))
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            model(ids[start:start + batch].to(device))
    for hook in hooks:
        hook.remove()
    return {name: s / ids.numel() for name, s in sums.items()}
