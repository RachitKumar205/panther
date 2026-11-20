"""Distillation: the block-scale parametrization and the top-k KL loss."""

from __future__ import annotations

import torch

from panther_quant.distill import BlockScale, kl_topk_loss


def test_block_scale_multiplies_each_256_block() -> None:
    w = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512)
    theta = torch.randn(4, 2)  # 512 / 256 = 2 blocks per row
    scale = BlockScale(4, 512)
    with torch.no_grad():
        scale.theta.copy_(theta)
        out = scale(w)
    for row in range(4):
        for block in range(2):
            expected = w[row, block * 256 : (block + 1) * 256] * torch.exp(theta[row, block])
            assert torch.allclose(out[row, block * 256 : (block + 1) * 256], expected)


def test_kl_topk_loss_is_nonnegative() -> None:
    logits = torch.randn(2, 3, 10)
    values, indices = torch.topk(torch.softmax(logits.float(), -1), 4, -1)
    assert kl_topk_loss(logits, values, indices) >= 0


def test_kl_topk_loss_rewards_matching_the_teacher() -> None:
    values = torch.tensor([[[0.9, 0.1, 0.0]]])  # teacher's renormalized top-3
    indices = torch.tensor([[[0, 1, 2]]])
    good = torch.tensor([[[5.0, 1.0, 0.0]]])  # student strongly favours token 0
    bad = torch.tensor([[[0.0, 1.0, 5.0]]])  # student strongly favours token 2
    assert kl_topk_loss(good, values, indices) < kl_topk_loss(bad, values, indices)
