"""Multiple-choice and GSM8K answers are computed correctly."""

from __future__ import annotations

import torch

from panther_quant.eval import completion_logprob, gsm8k_correct, multiple_choice


def test_completion_logprob_is_high_for_the_correct_next_token() -> None:
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[1, 2, 3, 4]])
    for t in range(3):
        logits[0, t, labels[0, t + 1]] = 100.0  # confident correct prediction
    assert completion_logprob(logits, labels) == torch.tensor(0.0) or completion_logprob(logits, labels) > -0.01


def test_multiple_choice_picks_the_most_likely_completion() -> None:
    vocab = 5
    wrong = (torch.zeros(1, 3, vocab), torch.tensor([[1, 1, 1]]))  # uniform -> low log-prob
    right_logits = torch.zeros(1, 3, vocab)
    right_labels = torch.tensor([[2, 2, 2]])
    for t in range(2):
        right_logits[0, t, 2] = 100.0
    assert multiple_choice([wrong, (right_logits, right_labels)]) == 1


def test_gsm8k_correct_matches_the_final_answer() -> None:
    assert gsm8k_correct("reasoning... #### 1,234", "1234")
    assert gsm8k_correct("#### 42.", "42")
    assert not gsm8k_correct("#### 42", "43")
    assert not gsm8k_correct("no answer here", "42")
