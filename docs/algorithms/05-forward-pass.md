# 05 — Forward pass and measurement

**Status:** derivation (forward pass + KL/PPL implemented; loader next)

## Objective

Compute next-token logits ourselves (ADR 0013), so KL, perplexity and answer flips come from our
own code and llama.cpp is only a parity oracle. The forward pass also produces the per-layer
activations GPTQ needs.

## Forward pass (Qwen2)

- **Embedding** `x = embed[ids]`, `(B, T, h)`.
- **Per layer**: `x += attention(RMSNorm(x))` then `x += mlp(RMSNorm(x))`.
  - RMSNorm: `x / sqrt(mean(x²) + eps) · w`.
  - Attention (grouped-query): project `q (heads·d)`, `k, v (kv_heads·d)`; apply **RoPE**
    (GPT-NeoX: `(x·cos + rotate_half(x)·sin)`); repeat `k, v` to `heads`; scaled dot-product with a
    causal mask; softmax in fp32; output projection. Qwen2 gives q/k/v a bias, o_proj none.
  - MLP (SwiGLU): `down(silu(gate(x)) · up(x))`.
- **Head**: final RMSNorm, then `lm_head` (the tied embedding when tied).
- Logits are upcast to **fp32** (ADR 0019); everything else runs bf16.

## Measurement (from logits)

- Cross-entropy: `mean_t -log p(x_{t+1} | x_≤t)`; perplexity `= exp(CE)`.
- KL: `mean_t KL(original ‖ ours) = Σ_v p_ref·(log p_ref − log p_ours)`.
- Same-top-word: fraction of positions where the two argmaxes agree (reported, not scored).
- Every position with a next token is scored (fixes llama.cpp's half-chunk behaviour, B4).

## Verification

- Forward pass: structural, determinism, causal-mask tests on synthetic weights (done).
- KL/PPL: perfect-model → perplexity 1; identical distributions → KL 0 (done).
- Parity: KL/PPL and logits vs llama.cpp on the same GGUF + text (when a real GGUF is available).
