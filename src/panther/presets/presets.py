"""llama.cpp's ready-made presets for one model: sized, built, and the competitor chosen for a budget."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from panther.fitness import Fitness
from panther.model import Model
from panther.model.model import DATA
from panther.presets.budget import Budget

# llama.cpp's presets, fewest bits first. Left out: Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL (lose to the K types at the
# same size), TQ1_0, TQ2_0 and MXFP4 (do not work on normal models), Q1_0 and Q2_0 (new, untested).
PRESETS = ("IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "Q2_K_S", "Q2_K", "IQ3_XXS", "IQ3_XS", "IQ3_S",
           "IQ3_M", "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0")


class Presets:
    """One model's presets, in data/models/<model>/presets/<PRESET>.gguf; shared by all its budgets."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.dir = model.dir / "presets"

    def file(self, preset: str) -> Path:
        return self.dir / f"{preset}.gguf"

    def dry_run(self, preset: str) -> str:
        """What llama-quantize --dry-run prints for the preset: each tensor's type and the total size, nothing built."""
        out = subprocess.run(["llama-quantize", "--dry-run", "--imatrix", self.model.imatrix, self.model.original,
                              preset], check=True, capture_output=True, text=True)
        return out.stdout + out.stderr

    def size(self, preset: str) -> int:
        """The preset's tensor bytes (exact)."""
        return int(float(re.search(r"quant size\s*=\s*([0-9.]+) MiB", self.dry_run(preset))[1]) * 2**20)

    def layout(self, preset: str) -> dict[str, str]:
        """The type the preset gives each matrix: {"blk.0.attn_q.weight": "IQ3_S", ...}."""
        return {name: type_.upper()
                for name, type_ in re.findall(r"\] +(\S+) +- .*\((\w+)\)$", self.dry_run(preset), re.M)}

    def build(self, preset: str) -> Path:
        """The preset file, made by llama-quantize from the original and the importance matrix."""
        if not self.file(preset).exists():
            self.dir.mkdir(exist_ok=True)
            part = self.dir / f"{preset}.part.gguf"
            subprocess.run(["llama-quantize", "--imatrix", self.model.imatrix, self.model.original, part, preset],
                           check=True)
            os.replace(part, self.file(preset))
        return self.file(preset)

    def best(self, budget: Budget, fitness: Fitness) -> dict:
        """The competitor for this budget: of the presets whose file fits, the one with the lowest fitness score
        (fitness: on the winner set's text and questions). If none fits, the smallest competes anyway, and the
        record says it is unfair (it has more RAM than we are allowed). Chosen once: saved to
        data/contests/<model>@<budget>/preset.json."""
        record = DATA / "contests" / f"{self.model.name}@{budget.gib:g}" / "preset.json"  # 5 and 5.0: @5
        if record.exists():
            return json.loads(record.read_text())
        sizes = {p: self.size(p) for p in PRESETS}
        fitting = [p for p in PRESETS if budget.fits(sizes[p]) and budget.fits(self.build(p).stat().st_size)]
        candidates = fitting or [min(PRESETS, key=sizes.get)]
        scores = {p: fitness.score(self.build(p)) for p in candidates}
        preset = min(scores, key=lambda p: scores[p].score)
        result = {"preset": preset, "file": str(self.file(preset)), "unfair": not fitting,
                  "ram_gib": budget.ram(self.file(preset).stat().st_size) / 2**30,
                  "scores": {p: asdict(s) for p, s in scores.items()}}
        record.parent.mkdir(parents=True, exist_ok=True)
        record.with_suffix(".part").write_text(json.dumps(result, indent=1))
        os.replace(record.with_suffix(".part"), record)
        return result

    def delete_unused(self) -> None:
        """Deletes every preset file no contest of this model chose."""
        records = (DATA / "contests").glob(f"{self.model.name}@*/preset.json")
        chosen = {json.loads(r.read_text())["preset"] for r in records}
        for f in self.dir.glob("*.gguf"):
            if f.stem not in chosen:
                f.unlink()
