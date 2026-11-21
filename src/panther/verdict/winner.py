"""The winner: every finalist's file, as the search built it and distilled (and QAT's, if it ran), scored on the
winner set; the lowest score wins."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from panther.distill import Distill
from panther.fitness import Fitness, Score
from panther.model.model import DATA
from panther.search import Genes


class Winner:
    """Chooses one contest's winner. The files live in data/contests/<contest>/ while it is chosen; only the
    winner's is kept."""

    def __init__(self, contest: str, genes: Genes, distill: Distill, fitness: Fitness) -> None:
        """contest: e.g. "Qwen2.5-14B-Instruct@5". fitness: on the winner set's text and questions."""
        self.genes, self.distill, self.fitness = genes, distill, fitness
        self.dir = DATA / "contests" / contest
        self.record = self.dir / "winner.json"

    def run(self, finalists: list[dict], preset: dict, extra: tuple[Path, ...] = ()) -> dict:
        """finalists: the search's (each with its genome). preset: part 2's record (its scores are on the winner set
        too). extra: more files that compete (QAT's, distilled). Returns the winner (its file, its score and its
        score relative to the preset's, every file's score), saved once to winner.json."""
        if self.record.exists():
            return json.loads(self.record.read_text())
        self.dir.mkdir(parents=True, exist_ok=True)
        files = []
        for i, finalist in enumerate(finalists):  # each finalist as the search built it, and distilled
            built = self.dir / f"finalist-{i}.gguf"
            if not built.exists():
                self.genes.build(finalist["genome"], built.with_name(built.name + ".part"))
                os.replace(built.with_name(built.name + ".part"), built)
            distilled = self.dir / f"finalist-{i}-distilled.gguf"
            if not distilled.exists():
                self.distill.run(built, distilled)
            files += [built, distilled]
        files += list(extra)
        with ThreadPoolExecutor(len(files)) as pool:  # every file scored at once (each run waits for its share)
            scores = dict(zip(files, pool.map(self.fitness.score, files), strict=True))
        best = min(scores, key=lambda f: scores[f].score)
        winner = self.dir / "winner.gguf"
        os.replace(best, winner)
        for f in files:  # only the winner's file is kept
            if f != best:
                f.unlink()
        relative = scores[best].relative(Score(**preset["scores"][preset["preset"]]))
        result = {"file": str(winner), "was": best.name, "score": asdict(scores[best]), "relative": relative,
                  "scores": {f.name: asdict(s) for f, s in scores.items()}}
        self.record.with_suffix(".part").write_text(json.dumps(result, indent=1))
        os.replace(self.record.with_suffix(".part"), self.record)
        return result
