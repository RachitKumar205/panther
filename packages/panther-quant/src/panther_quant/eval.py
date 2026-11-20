"""Benchmark answers from logits: multiple-choice and GSM8K (the flips half of the score)."""

from __future__ import annotations

import re

import torch


def completion_logprob(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean log-prob of a completion: position ``t`` predicts ``labels[t+1]``."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[:, :-1].gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1).mean())


def multiple_choice(choices: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Index of the choice whose completion the model finds most likely."""
    return max(range(len(choices)), key=lambda i: completion_logprob(*choices[i]))


def gsm8k_correct(completion: str, answer: str) -> bool:
    """Whether the completion ends with ``#### answer`` matching the reference (GSM8K)."""
    match = re.search(r"#### (-?[0-9.,]+)", completion)
    return match is not None and _norm(match.group(1)) == _norm(answer)


def _norm(value: str) -> str:
    return re.sub(r"[$,\s]", "", value).rstrip(".")
