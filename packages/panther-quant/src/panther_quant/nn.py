"""The Qwen2 forward pass, from first principles (algorithm 05).

Runs in bfloat16; logits are upcast to float32 before the softmax (ADR 0019). This is the
measurement engine and the source of per-layer activations for GPTQ.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: ``x / sqrt(mean(x^2) + eps) * weight``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # per-channel gain
        self.eps = eps  # stability term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """The 90-degree rotation of RoPE: swap halves and negate the first."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope(
    head_dim: int,
    seq_len: int,
    theta: float = 1_000_000.0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin tables of shape ``(1, 1, seq_len, head_dim)`` (GPT-NeoX layout)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype).unsqueeze(0).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the rotary position embedding to q and k (broadcast over batch and heads)."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    """Grouped-query attention with RoPE; Qwen2 gives q/k/v a bias, o_proj none."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads  # query/output heads
        self.kv_heads = kv_heads  # grouped-query key/value heads
        self.head_dim = head_dim  # per-head width
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)  # match query heads
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)  # fp32 softmax, back to bf16
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    """One decoder block: attention and MLP, each pre-normed and residual."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = Attention(hidden, heads, kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = MLP(hidden, intermediate)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    """The Qwen2 decoder: embed -> layers -> final norm -> lm_head."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size  # model width
        self.head_dim = head_dim  # per-head width
        self.rope_theta = rope_theta  # RoPE base
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            Layer(hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size)
            for _ in range(num_hidden_layers)
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, seq_len, vocab_size)`` in float32."""
        t = input_ids.shape[1]
        cos, sin = precompute_rope(self.head_dim, t, self.rope_theta, self.embed_tokens.weight.dtype)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=input_ids.device), diagonal=1)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.lm_head(self.norm(x)).float()  # fp32 logits (ADR 0019)
