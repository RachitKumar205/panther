"""KL, perplexity and same-top-word from logits."""

from __future__ import annotations

import math

import pytest
import torch

from panther_quant.measure import cross_entropy, kl_divergence, perplexity, same_top_word


def test_perplexity_is_one_for_a_perfect_model() -> None:
    labels = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 5)
    for t in range(3):  # position t predicts labels[t+1]
        logits[0, t, labels[0, t + 1]] = 100.0
    assert perplexity(logits, labels) == pytest.approx(1.0, abs=1e-3)


def test_cross_entropy_is_log_vocab_for_uniform_logits() -> None:
    labels = torch.zeros((1, 4), dtype=torch.long)
    logits = torch.zeros(1, 4, 8)  # vocab 8, uniform -> CE = log(8)
    assert cross_entropy(logits, labels) == pytest.approx(math.log(8))


def test_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.randn(1, 4, 10)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-5)


def test_kl_is_nonnegative() -> None:
    assert kl_divergence(torch.randn(2, 5, 10), torch.randn(2, 5, 10)) >= 0


def test_same_top_word_counts_matching_argmax() -> None:
    a = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])  # argmax 1, 0
    b = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])  # argmax 1, 1
    assert same_top_word(a, b) == pytest.approx(50.0)
