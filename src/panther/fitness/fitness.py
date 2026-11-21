"""The fitness of a file: how differently it behaves from the original. score = flips % + 10 × KL, lower is better."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np

from panther.fitness.resources import GIB, Resources
from panther.model import Model

KL_WEIGHT = 10
KL_CONTEXT = 2048  # KL reads the text in chunks of 2,048 tokens and scores the second half of each
TASK_CONTEXT, TASK_SEQUENCES = 4096, 5  # multiple choice and winogrande: 5 sequences of 4,096 tokens at once


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


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


class Fitness:
    """Fitness on one text and one set of questions: a task name → its llama.cpp question file (a binary file for
    multiple choice, a CSV for winogrande). The original's answers are recorded once, in data/models/<model>/
    reference/, named by the content of the text or question file they were made from.

    A file's measurements (KL and one per task) run at the same time, and several files can be scored at once: every
    llama-perplexity run waits for its share of GPU memory and RAM (resources)."""

    def __init__(self, model: Model, text: Path, questions: dict[str, Path]) -> None:
        self.model, self.text, self.questions = model, text, questions
        self.reference = model.dir / "reference"
        self.resources = Resources()
        self.vocabulary = int(next(t for t in gguf.GGUFReader(model.original).tensors
                                   if t.name == "token_embd.weight").shape[1])
        self.recording = threading.Lock()  # the original's records are made once, by one caller

    def score(self, file: Path) -> Score:
        with self.recording:
            self.kl_base()
            original = {task: self.original_answers(task) for task in self.questions}
        with ThreadPoolExecutor(1 + len(self.questions)) as pool:
            kl = pool.submit(self.kl, file)
            answers = {task: pool.submit(self.answers, file, task) for task in self.questions}
            flips = {task: 100 * float(np.mean(a.result() != original[task])) for task, a in answers.items()}
            kl, same_top = kl.result()
        mean_flips = float(np.mean(list(flips.values())))
        return Score(mean_flips + KL_WEIGHT * kl, mean_flips, kl, same_top, flips)

    def perplexity(self, file: Path, positions: int, *args) -> str:
        """Runs llama-perplexity on a GGUF and returns everything it printed. It holds the file plus 10 % and 4 GiB
        of GPU memory, and its next-word probabilities for every position it reads at once (float32) plus 1 GiB of
        RAM (measured on the 1.5B: 12.6 GB for 5 sequences of 4,096 positions)."""
        size = file.stat().st_size
        with self.resources.hold(int(1.1 * size) + 4 * GIB, positions * self.vocabulary * 4 + GIB):
            run = subprocess.run(["llama-perplexity", "-m", file, "-ngl", "99", *args], check=True,
                                 capture_output=True, text=True)
        return run.stdout + run.stderr

    def kl(self, file: Path) -> tuple[float, float]:
        """(mean KL, same top word %) against the original's next-word probabilities on the text."""
        return self.kl_chunks(file)[:2]

    def kl_chunks(self, file: Path) -> tuple[float, float, np.ndarray]:
        """(mean KL, same top word %, each chunk's KL). llama-perplexity prints, after each chunk, the mean KL of the
        chunks so far (m_i): chunk i's own KL is i·m_i − (i−1)·m_(i−1). Its 5 decimals make each chunk's KL a little
        noisy, which can only widen an error bar made from them."""
        out = self.perplexity(file, KL_CONTEXT, "--kl-divergence-base", self.kl_base(), "--kl-divergence",
                              "-c", str(KL_CONTEXT), "-b", str(KL_CONTEXT))
        # chunk, PPL ± x, ln(PPL ratio) ± x, then the running mean KL
        running = np.array([float(m) for m in re.findall(
            r"^\s*\d+\s+[0-9.]+\s+±\s+[0-9.]+\s+-?[0-9.]+\s+±\s+[0-9.]+\s+(-?[0-9.]+)\s+±", out, re.M)])
        chunks = int(re.search(r"computing over (\d+) chunks", out)[1])
        assert len(running) == chunks, f"{file.name}: KL for {len(running)} of {chunks} chunks"
        i = np.arange(1, chunks + 1)
        each = i * running - np.concatenate([[0.0], (i[:-1]) * running[:-1]])
        return (float(re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)", out)[1]),
                float(re.search(r"Same top p:\s*([0-9.]+)", out)[1]), each)

    def kl_base(self) -> Path:
        """The original's next-word probabilities on the text, recorded once."""
        base = self.reference / f"{digest(self.text)}.kld"
        if not base.exists():
            self.reference.mkdir(parents=True, exist_ok=True)
            part = base.with_suffix(".part")
            self.perplexity(self.model.original, KL_CONTEXT, "-f", self.text, "-c", str(KL_CONTEXT),
                            "-b", str(KL_CONTEXT), "--kl-divergence-base", part)
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
            out = self.perplexity(file, TASK_SEQUENCES * TASK_CONTEXT, "-f", questions, "--winogrande",
                                  "-c", str(TASK_CONTEXT), "-np", str(TASK_SEQUENCES))
            rows = re.findall(r"^\d+\t[0-9.]+\t\s*\S+\s+\S+\s+([12])\s+([12])$", out, re.M)  # chosen, correct
            right = np.array([chosen == correct for chosen, correct in rows])
            count = len(questions.read_text().strip().split("\n"))
        else:
            out = self.perplexity(file, TASK_SEQUENCES * TASK_CONTEXT, "-bf", questions, "--multiple-choice",
                                  "-c", str(TASK_CONTEXT), "-np", str(TASK_SEQUENCES))
            accuracy = np.array([float(a) for a in re.findall(r"^\d+\t(\d+\.\d{8})$", out, re.M)])  # running, %
            right_so_far = np.round(accuracy * np.arange(1, len(accuracy) + 1) / 100).astype(int)
            right = np.diff(right_so_far, prepend=0).astype(bool)
            count = int.from_bytes(questions.read_bytes()[:4], "little")  # the binary file starts with its count
        assert len(right) == count, f"{file.name}, {task}: answers for {len(right)} of {count} questions"
        return right
