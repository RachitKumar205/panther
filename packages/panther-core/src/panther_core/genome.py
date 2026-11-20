"""Genomes: one quantization type per matrix, kept inside a RAM window.

The operators here are the pure, deterministic core of the genetic search. An injected
:class:`random.Random` makes runs reproducible (bug B8); every mutation, crossover and repair
returns an assignment inside the window.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

from panther_core.geometry import ModelGeometry
from panther_core.quant import QuantType
from panther_core.ram import RamModel

Assignment = dict[str, QuantType]


@dataclass(frozen=True, slots=True)
class RamWindow:
    """An inclusive byte range a genome's total RAM must fall in."""

    lo: int
    hi: int

    def contains(self, ram: int) -> bool:
        return self.lo <= ram <= self.hi


class Genome:
    """One-type-per-matrix allocation over a model, with the search operators."""

    def __init__(
        self,
        geometry: ModelGeometry,
        ram_model: RamModel,
        allowed: Mapping[str, tuple[QuantType, ...]],
        window: RamWindow,
        error: Mapping[str, Mapping[QuantType, float]],
        rng: random.Random | None = None,
    ) -> None:
        self.geometry = geometry
        self.ram_model = ram_model
        self.allowed = dict(allowed)
        self.window = window
        self.error = error
        self.rng = rng or random.Random()
        by_name = {matrix.name: matrix for matrix in geometry.matrices}
        self.bytes = {n: {t: by_name[n].nbytes(t) for t in ts} for n, ts in self.allowed.items()}
        self.group = {
            n: (by_name[n].role, by_name[n].layer * 4 // geometry.spec.num_hidden_layers)
            for n in self.allowed
        }

    def ram(self, assignment: Mapping[str, QuantType]) -> int:
        return self.ram_model.total_bytes(assignment)

    def fits(self, assignment: Mapping[str, QuantType]) -> bool:
        return self.window.contains(self.ram(assignment))

    def uniform(self, quant: QuantType) -> Assignment:
        return {name: quant for name in self.allowed}

    def _move(self, assignment: Assignment, name: str, delta: int) -> None:
        types = self.allowed[name]
        i = max(0, min(len(types) - 1, types.index(assignment[name]) + delta))
        assignment[name] = types[i]

    def repair(self, assignment: Mapping[str, QuantType]) -> Assignment:
        """Step matrices one at a time until inside the window, least error added per byte."""
        out = dict(assignment)
        for _ in range(20_000):
            ram = self.ram(out)
            if self.window.contains(ram):
                return out
            step = -1 if ram > self.window.hi else 1
            gains: list[tuple[float, str]] = []
            for name, types in self.allowed.items():
                i = types.index(out[name]) + step
                if not 0 <= i < len(types):
                    continue
                new, db = types[i], self.bytes[name][types[i]] - self.bytes[name][out[name]]
                if step < 0 or ram + db <= self.window.hi:
                    gain = (self.error[name][new] - self.error[name][out[name]]) / max(1, abs(db))
                    gains.append((gain, name))
            if not gains:
                return out
            self._move(out, self.rng.choice(sorted(gains)[:8])[1], step)
        return out

    def mutate(self, assignment: Mapping[str, QuantType]) -> Assignment:
        out = dict(assignment)
        names, x = list(self.allowed), self.rng.random()
        if x < 0.8:
            for name in self.rng.sample(names, self.rng.randint(1, 3)):
                self._move(out, name, self.rng.choice((-1, 1) if x < 0.5 else (-3, -2, 2, 3)))
        else:
            group, delta = self.rng.choice(sorted(set(self.group.values()))), self.rng.choice((-1, 1))
            for name in names:
                if self.group[name] == group:
                    self._move(out, name, delta)
        return self.repair(out)

    def crossover(self, p: Mapping[str, QuantType], q: Mapping[str, QuantType]) -> Assignment:
        take = {layer: self.rng.random() < 0.5 for layer in {m.layer for m in self.geometry.matrices}}
        by_name = {m.name: m for m in self.geometry.matrices}
        return self.repair({n: (p[n] if take[by_name[n].layer] else q[n]) for n in p})
