"""The verdict: the winner against the preset, once, on the verdict set. Relative score = ours ÷ the preset's × 100
(score = flips % + 10 × KL, as the fitness), with a 95 % interval: WIN, TIE or LOSE."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panther.fitness import KL_WEIGHT, Fitness
from panther.model.model import DATA
from panther.verdict.gsm8k import GSM8K

Z = 1.96  # 95 %: the true value is inside ±1.96 standard errors 19 times in 20


@dataclass
class Measured:
    """One file on the verdict set."""

    right: dict[str, np.ndarray]  # task → True/False per question
    kl: float
    same_top: float
    chunks: np.ndarray  # each text chunk's KL


def judge(ours: Measured, preset: Measured, original: dict[str, np.ndarray], weight: float = KL_WEIGHT) -> dict:
    """Our relative score against the preset's and its 95 % interval, from the standard error of the difference of
    the two scores (the same questions and text for both, so the difference is paired):
      flips: per task, each question adds (our flip − the preset's flip), which is 0 where both or neither flipped;
             the standard error of each task's mean, combined over the tasks as their mean is;
      KL:    each chunk's (our KL − the preset's KL); the standard error of their mean;
    the two together as independent. WIN if the whole interval is below 100, LOSE if it is all above, else TIE."""
    tasks = list(original)
    ours_flips = {t: ours.right[t] != original[t] for t in tasks}
    preset_flips = {t: preset.right[t] != original[t] for t in tasks}
    ours_score = 100 * np.mean([ours_flips[t].mean() for t in tasks]) + weight * ours.kl
    preset_score = 100 * np.mean([preset_flips[t].mean() for t in tasks]) + weight * preset.kl
    differences = {t: ours_flips[t].astype(float) - preset_flips[t].astype(float) for t in tasks}
    flips_se = 100 * np.sqrt(sum(d.var(ddof=1) / len(d) for d in differences.values())) / len(tasks)
    gap = ours.chunks - preset.chunks
    kl_se = gap.std(ddof=1) / np.sqrt(len(gap))
    se = float(np.hypot(flips_se, weight * kl_se))
    relative = 100 * ours_score / preset_score
    low, high = 100 * (ours_score - Z * se) / preset_score, 100 * (ours_score + Z * se) / preset_score
    return {"relative": float(relative), "interval": [float(low), float(high)],
            "verdict": "WIN" if high < 100 else "LOSE" if low > 100 else "TIE",
            "score": float(ours_score), "preset_score": float(preset_score), "standard_error": se}


class Verdict:
    """The verdict set: its text and multiple-choice/winogrande questions (fitness), and its GSM8K questions."""

    def __init__(self, fitness: Fitness, gsm8k: GSM8K) -> None:
        self.fitness, self.gsm8k = fitness, gsm8k

    def measure(self, file: Path) -> Measured:
        """Every question and the text, all of a file's runs at once."""
        with ThreadPoolExecutor(2 + len(self.fitness.questions)) as pool:
            kl = pool.submit(self.fitness.kl_chunks, file)
            right = {t: pool.submit(self.fitness.answers, file, t) for t in self.fitness.questions}
            right["gsm8k"] = pool.submit(self.gsm8k.answers, file)
            right = {t: r.result() for t, r in right.items()}
            mean, same_top, chunks = kl.result()
        return Measured(right, mean, same_top, chunks)

    def run(self, contest: str, winner: Path, preset: dict) -> dict:
        """The verdict for a contest (e.g. "Qwen2.5-14B-Instruct@5"), saved once to data/contests/<contest>/
        verdict.json. preset: part 2's record (its file, and whether it is unfair)."""
        record = DATA / "contests" / contest / "verdict.json"
        if record.exists():
            return json.loads(record.read_text())
        original = {t: self.fitness.original_answers(t) for t in self.fitness.questions}
        original["gsm8k"] = self.gsm8k.original_answers()
        with ThreadPoolExecutor(2) as pool:  # the winner and the preset at the same time
            ours, theirs = pool.map(self.measure, [winner, Path(preset["file"])])
        result = {**judge(ours, theirs, original), "unfair": preset["unfair"], "preset": preset["preset"],
                  "kl_weight_5": judge(ours, theirs, original, 5)["verdict"],
                  "kl_weight_20": judge(ours, theirs, original, 20)["verdict"],
                  "accuracy": {who: {t: float(100 * m.right[t].mean()) for t in m.right}
                               for who, m in (("ours", ours), ("preset", theirs))}
                  | {"original": {t: float(100 * r.mean()) for t, r in original.items()}},
                  "kl": {"ours": ours.kl, "preset": theirs.kl},
                  "same_top": {"ours": ours.same_top, "preset": theirs.same_top}}
        kept = {who: 100 * np.mean(list(result["accuracy"][who].values()))
                / np.mean(list(result["accuracy"]["original"].values())) for who in ("ours", "preset")}
        result["accuracy_kept"] = {who: float(v) for who, v in kept.items()}
        record.parent.mkdir(parents=True, exist_ok=True)
        record.with_suffix(".part").write_text(json.dumps(result, indent=1))
        os.replace(record.with_suffix(".part"), record)
        return result
