"""Distillation: block-scale multipliers and the top-k KL loss (algorithm 09)."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from panther_quant.nn import Layer, Qwen2Model

# Byte offset(s) of each type's 16-bit block scale(s) inside one block, for the distillation write-back.
D_OFF = {
    "Q2_K": (80, 82),
    "Q3_K": (108,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q6_K": (208,),
    "Q8_0": (0,),
    "IQ1_S": (0,),
    "IQ2_XXS": (0,),
    "IQ2_XS": (0,),
    "IQ2_S": (0,),
    "IQ3_XXS": (0,),
    "IQ3_S": (0,),
    "IQ4_XS": (0,),
}


class BlockScale(nn.Module):
    """Multiply each 256-weight block of a weight by ``exp(theta)``, ``theta`` in fp32."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(out_features, in_features // 256))  # one multiplier per block

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scaled = w.float().view(w.shape[0], -1, 256) * torch.exp(self.theta)[:, :, None]
        return scaled.view(w.shape).to(w.dtype)


def kl_topk_loss(
    logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(teacher top-k || student) over positions.

    Args:
        logits: Student logits ``(..., vocab)``.
        teacher_values: Teacher probabilities ``(..., k)``, renormalized over the top-k.
        teacher_indices: Teacher token ids ``(..., k)``.
    """
    log_ps = torch.log_softmax(logits.float(), dim=-1)  # (..., vocab)
    p_t = teacher_values.float() / teacher_values.float().sum(-1, keepdim=True)
    log_p_t = torch.log(p_t + 1e-12)  # avoid log(0) -> -inf
    gathered = log_ps.gather(-1, teacher_indices.long())  # (..., k)
    return (p_t * (log_p_t - gathered)).sum(-1).mean()


def passes(n: int, batch: int, steps: int, seed: int = 0) -> list[torch.Tensor]:
    """``steps`` batches of ``batch`` sequence indices; each sequence used once before any repeat."""
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(n) for _ in range(-(-steps * batch // n))])
    return [torch.tensor(order[k * batch : (k + 1) * batch]) for k in range(steps)]


def teacher_tops(model: Qwen2Model, ids: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The teacher's top-k next-word probabilities and ids per position: ``(probs, indices)``."""
    with torch.no_grad():
        top = torch.topk(torch.log_softmax(model(ids).float(), -1), k, -1)
    return torch.softmax(top.values, -1), top.indices


def distill(
    student: Qwen2Model,
    teacher: Qwen2Model,
    ids: torch.Tensor,
    steps: int = 600,
    batch: int = 2,
    topk: int = 64,
    lr_theta: float = 2e-3,
    lr_norm: float = 1e-4,
    seed: int = 0,
) -> list[float]:
    """Train the student's block scales and norms against the teacher's top-k; return the loss history."""
    with torch.no_grad():
        values, indices = teacher_tops(teacher, ids, topk)
    theta_params, norm_params = _parametrize(student)
    optimizer = torch.optim.Adam([{"params": theta_params, "lr": lr_theta}, {"params": norm_params, "lr": lr_norm}])
    losses: list[float] = []
    for batch_ids in passes(ids.shape[0], batch, steps, seed):
        loss = kl_topk_loss(student(ids[batch_ids]), values[batch_ids], indices[batch_ids])
        optimizer.zero_grad()
        loss.backward()  # type: ignore[reportUnknownMemberType]
        optimizer.step()  # type: ignore[reportUnknownMemberType]
        losses.append(float(loss.detach()))
    return losses


def _parametrize(student: Qwen2Model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Register a :class:`BlockScale` on every layer matrix, collecting theta and norm parameters."""
    theta_params: list[nn.Parameter] = []
    norm_params: list[nn.Parameter] = []
    for layer in student.layers:
        layer = cast(Layer, layer)
        attn, mlp = layer.self_attn, layer.mlp
        for module in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if module.weight.shape[1] >= 256:  # one theta per 256 input columns
                scale = BlockScale(module.weight.shape[0], module.weight.shape[1])
                nn.utils.parametrize.register_parametrization(module, "weight", scale)
                theta_params.append(scale.theta)
        norm_params += [layer.input_layernorm.weight, layer.post_attention_layernorm.weight]
    norm_params.append(student.norm.weight)
    return theta_params, norm_params
