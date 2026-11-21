"""The fitness of a file: how differently it behaves from the original. score = flips % + 10 × KL, lower is better."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panther.model import Model

KL_WEIGHT = 10
KL_CONTEXT = 2048  # KL reads the text in chunks of 2,048 tokens and scores the second half of each


@dataclass(frozen=True)
class Score:
    score: float  # flips + KL_WEIGHT × kl
    flips: float  # % of questions whose right/wrong differs from the original's, the mean over the tasks
    kl: float  # mean KL divergence from the original's next-word probabilities over the text's scored tokens
    same_top: float  # % of tokens where the most likely next word is the original's (reported, not scored)
    flips_per_task: dict[str, float]

    def relative(self, preset: Score) -> float:
        """This score as % of the preset's on the same questions and text: 100 = the preset's damage, 0 = none."""
        return 100 * self.score / preset.score


def perplexity(file: Path, *args) -> str:
    """Runs llama-perplexity on a GGUF and returns everything it printed."""
    run = subprocess.run(["llama-perplexity", "-m", file, "-ngl", "99", *args], check=True, capture_output=True,
                         text=True)
    return run.stdout + run.stderr


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


class Fitness:
    """Fitness on one text and one set of questions: a task name → its llama.cpp question file (a binary file for
    multiple choice, a CSV for winogrande). The original's answers are recorded once, in data/models/<model>/
    reference/, named by the content of the text or question file they were made from."""

    def __init__(self, model: Model, text: Path, questions: dict[str, Path]) -> None:
        self.model, self.text, self.questions = model, text, questions
        self.reference = model.dir / "reference"

    def score(self, file: Path) -> Score:
        flips = {task: 100 * float(np.mean(self.answers(file, task) != self.original_answers(task)))
                 for task in self.questions}
        kl, same_top = self.kl(file)
        mean_flips = float(np.mean(list(flips.values())))
        return Score(mean_flips + KL_WEIGHT * kl, mean_flips, kl, same_top, flips)

    def kl(self, file: Path) -> tuple[float, float]:
        """(mean KL, same top word %) against the original's next-word probabilities on the text."""
        out = perplexity(file, "--kl-divergence-base", self.kl_base(), "--kl-divergence",
                         "-c", str(KL_CONTEXT), "-b", str(KL_CONTEXT))
        return (float(re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)", out)[1]),
                float(re.search(r"Same top p:\s*([0-9.]+)", out)[1]))

    def kl_base(self) -> Path:
        """The original's next-word probabilities on the text, recorded once."""
        base = self.reference / f"{digest(self.text)}.kld"
        if not base.exists():
            self.reference.mkdir(parents=True, exist_ok=True)
            part = base.with_suffix(".part")
            perplexity(self.model.original, "-f", self.text, "-c", str(KL_CONTEXT), "-b", str(KL_CONTEXT),
                       "--kl-divergence-base", part)
            os.replace(part, base)
        return base

    def original_answers(self, task: str) -> np.ndarray:
        """The original's right/wrong on a task's questions, recorded once."""
        record = self.reference / f"{digest(self.questions[task])}.json"
        if not record.exists():
            self.reference.mkdir(parents=True, exist_ok=True)
            record.with_suffix(".part").write_text(json.dumps(self.answers(self.model.original, task).tolist()))
            os.replace(record.with_suffix(".part"), record)
        return np.array(json.loads(record.read_text()))

    def answers(self, file: Path, task: str) -> np.ndarray:
        """True/False per question, in file order, for every question of the task's file."""
        questions = self.questions[task]
        if task == "winogrande":
            out = perplexity(file, "-f", questions, "--winogrande", "-c", "4096", "-np", "5")
            rows = re.findall(r"^\d+\t[0-9.]+\t\s*\S+\s+\S+\s+([12])\s+([12])$", out, re.M)  # chosen, correct
            right = np.array([chosen == correct for chosen, correct in rows])
            count = len(questions.read_text().strip().split("\n"))
        else:
            out = perplexity(file, "-bf", questions, "--multiple-choice", "-c", "4096", "-np", "5")
            accuracy = np.array([float(a) for a in re.findall(r"^\d+\t(\d+\.\d{8})$", out, re.M)])  # running, %
            right_so_far = np.round(accuracy * np.arange(1, len(accuracy) + 1) / 100).astype(int)
            right = np.diff(right_so_far, prepend=0).astype(bool)
            count = int.from_bytes(questions.read_bytes()[:4], "little")  # the binary file starts with its count
        assert len(right) == count, f"{file.name}, {task}: answers for {len(right)} of {count} questions"
        return right
