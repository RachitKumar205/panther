"""Distillation: passes use each sequence once; training reduces the loss."""

from __future__ import annotations

import torch

from panther_quant.distill import distill, passes, teacher_tops
from panther_quant.nn import Qwen2Model


def build() -> Qwen2Model:
    return Qwen2Model(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        intermediate_size=512,
        vocab_size=100,
    )


def test_passes_uses_each_sequence_once_per_pass() -> None:
    order = [int(i) for batch in passes(6, 2, 6, seed=0) for i in batch]
    assert sorted(order) == sorted([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])


def test_teacher_tops_has_the_right_shape_and_is_normalized() -> None:
    ids = torch.randint(0, 100, (2, 8))
    values, indices = teacher_tops(build().eval(), ids, k=8)
    assert values.shape == (2, 8, 8)
    assert indices.shape == (2, 8, 8)
    assert torch.allclose(values.sum(-1), torch.ones(2, 8))


def test_distill_reduces_the_loss() -> None:
    teacher = build().eval()
    student = build()
    student.load_state_dict(teacher.state_dict())  # identical, theta = 0
    ids = torch.randint(0, 100, (2, 8))
    losses = distill(student, teacher, ids, steps=10, batch=2, topk=8, seed=0)
    assert losses[-1] < losses[0]
