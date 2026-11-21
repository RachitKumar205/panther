"""Genomes, one type per matrix: their exact size, their file, and the search's moves (slice, mutate, repair)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import gguf
import numpy as np
from ortools.sat.python import cp_model

from panther.model import TYPES, Model
from panther.model.model import NEEDS_IMPORTANCE, bits
from panther.presets import Budget
from panther.presets.budget import GIB, TOP

BOTTOM = 0.99  # a genome must use at least 0.99 × the budget (the top, 1.005 ×, is Budget's)

Genome = dict[str, str]  # matrix name → type
ROLES = ("token_embd", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "output")
RIDGE = 10  # how strongly the solver's weights are pulled toward 1: as strongly as 10 scored genomes


def role(matrix: str) -> str:
    """blk.3.ffn_up.weight → ffn_up; token_embd.weight → token_embd; output.weight → output."""
    return matrix.split(".")[2] if matrix.startswith("blk.") else matrix.split(".")[0]


def position(matrix: str) -> tuple[int, int]:
    """Where a matrix runs in the model: the word table first, then layer by layer, the output layer last."""
    if matrix == "token_embd.weight":
        return 0, 0
    if matrix == "output.weight":
        return 2, 0
    return 1, int(matrix.split(".")[1])


class Genes:
    """What every genome of one model at one budget is made of: the matrices in the order the model runs them, the
    types each may take, their exact bytes and output errors (errors.json, from part 1)."""

    def __init__(self, model: Model, budget: Budget) -> None:
        self.model, self.budget = model, budget
        original = gguf.GGUFReader(model.original)
        matrices = sorted((t for t in original.tensors if len(t.shape) == 2), key=lambda t: position(t.name))
        self.order = [t.name for t in matrices]
        self.units = [[t.name for t in matrices if position(t.name) == p]  # crossover moves whole units
                      for p in sorted({position(t.name) for t in matrices})]
        self.allowed = {m: [t for t in TYPES if not (m == "token_embd.weight" and t in NEEDS_IMPORTANCE)]
                        for m in self.order}

        def stored(byte_shape: tuple[int, ...]) -> int:  # a tensor's bytes in the file, padded to the alignment
            n = int(byte_shape[0]) * int(byte_shape[1])
            return -(-n // original.alignment) * original.alignment

        shapes = {t.name: (int(t.shape[1]), int(t.shape[0])) for t in matrices}  # rows, columns
        self.bytes = {m: {t: stored(gguf.quant_shape_to_byte_shape(shapes[m], gguf.GGMLQuantizationType[t]))
                          for t in self.allowed[m]} for m in self.order}
        # everything that is not a matrix (metadata, norms, biases): the same in every genome's file
        self.rest = model.original.stat().st_size - sum(stored(t.data.shape) for t in matrices)
        self.errors = json.loads(model.errors.read_text())

    def size(self, genome: Genome) -> int:
        """The genome's file size, exactly, before it is built."""
        return self.rest + sum(self.bytes[m][genome[m]] for m in self.order)

    def fits(self, genome: Genome) -> bool:
        size = self.size(genome)
        return self.budget.fits(size) and self.budget.ram(size) >= BOTTOM * self.budget.gib * GIB

    def solve(self, count: int, weights: dict[str, float] | None = None, away: list[Genome] = (),
              distance: int = 3) -> list[Genome]:
        """Up to count genomes inside the window with the lowest weighted summed output error (errors.json, each
        matrix's error times its role's weight, all 1 by default), each at least `distance` matrices different from
        the others and from every genome in away. By CP-SAT (OR-Tools), on one thread, so the same question gets the
        same answer."""
        weights = weights or dict.fromkeys(ROLES, 1.0)
        model = cp_model.CpModel()
        choose = {(m, t): model.new_bool_var(f"{m}:{t}") for m in self.order for t in self.allowed[m]}
        for m in self.order:
            model.add_exactly_one(choose[m, t] for t in self.allowed[m])
        room = self.budget.kv_cache + self.budget.overhead + self.rest  # everything but the matrices
        low = math.ceil(BOTTOM * self.budget.gib * GIB) - room  # rounded inward: every answer fits()
        high = math.floor(TOP * self.budget.gib * GIB) - room
        model.add_linear_constraint(sum(self.bytes[m][t] * x for (m, t), x in choose.items()), low, high)
        cost = {(m, t): weights[role(m)] * self.errors[m][t] for m, t in choose}
        top = max(cost.values())
        model.minimize(sum(round(1e9 * cost[k] / top) * x for k, x in choose.items()))
        for genome in away:
            model.add(sum(choose[m, t] for m, t in genome.items()) <= len(self.order) - distance)
        found: list[Genome] = []
        for _ in range(count):
            solver = cp_model.CpSolver()
            solver.parameters.num_workers, solver.parameters.max_deterministic_time = 1, 60
            if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            found.append({m: t for (m, t), x in choose.items() if solver.value(x)})
            model.add(sum(choose[m, t] for m, t in found[-1].items()) <= len(self.order) - distance)
        return found

    def fit(self, genomes: list[Genome], scores: list[float]) -> dict[str, float]:
        """One weight per role, so that score ≈ α × Σ (weight × summed output error of the role's matrices) + c over
        the scored genomes. A ridge fit, every weight pulled toward 1 with the strength of RIDGE scored genomes: with
        few scores the weights stay near 1 (the plain summed error), and move only as far as the scores support.
        Weights stay positive."""
        if len(genomes) < 2:
            return dict.fromkeys(ROLES, 1.0)
        x = np.array([[sum(self.errors[m][g[m]] for m in self.order if role(m) == r) for r in ROLES] for g in genomes])
        y = np.array(scores)
        alpha, c = np.polyfit(x.sum(axis=1), y, 1)  # first: the score against the plain summed error
        if alpha <= 0:
            return dict.fromkeys(ROLES, 1.0)
        a = alpha * x  # each role's contribution to the score at weight 1
        # every role pulled toward 1 equally, in score units: a role that barely changes the score stays near 1
        prior = RIDGE * (a**2).mean() * np.eye(len(ROLES))
        w = 1 + np.linalg.solve(a.T @ a + prior, a.T @ (y - c - a.sum(axis=1)))
        return dict(zip(ROLES, np.maximum(w, 0.01).tolist(), strict=True))

    def nearest(self, matrix: str, type_: str) -> str:
        """The type this matrix may take that is closest in bits to type_ (e.g. IQ3_S, not one of ours: Q3_K)."""
        return min(self.allowed[matrix], key=lambda t: abs(bits(t) - bits(type_)))

    def like(self, layout: dict[str, str]) -> Genome:
        """A genome as close as our types allow to a layout (a preset's, or one type everywhere)."""
        return {m: self.nearest(m, layout[m]) for m in self.order}

    def slice(self, a: Genome, b: Genome, rng: random.Random) -> Genome:
        """Two-point crossover: the units between two cut points from b, the rest from a. A unit is the word table, one
        layer's 7 matrices, or the output layer, so a layer always comes from one parent."""
        i, j = sorted(rng.sample(range(len(self.units) + 1), 2))
        child = dict(a)
        for unit in self.units[i:j]:
            child.update({m: b[m] for m in unit})
        return child

    def mutate(self, genome: Genome, rng: random.Random) -> Genome:
        """1 to 3 random matrices, each 1 or 2 types up or down."""
        child = dict(genome)
        for m in rng.sample(self.order, rng.randint(1, 3)):
            types = self.allowed[m]
            child[m] = types[min(max(types.index(child[m]) + rng.choice((-2, -1, 1, 2)), 0), len(types) - 1)]
        return child

    def repair(self, genome: Genome, rng: random.Random) -> Genome:
        """Into the window, one matrix one type at a time: too big, step down one of the 8 matrices that add the least
        output error per byte saved; too small, step up one of the 8 that remove the most per byte added (without
        going over the budget)."""
        genome = dict(genome)
        for _ in range(10_000):
            if self.fits(genome):
                return genome
            size = self.size(genome)
            step = 1 if self.budget.fits(size) else -1
            moves = []
            for m in self.order:
                i = self.allowed[m].index(genome[m]) + step
                if not 0 <= i < len(self.allowed[m]):
                    continue
                new = self.allowed[m][i]
                change = self.bytes[m][new] - self.bytes[m][genome[m]]
                if step == 1 and not self.budget.fits(size + change):
                    continue
                moves.append(((self.errors[m][new] - self.errors[m][genome[m]]) / abs(change), m, new))
            assert moves, "no matrix can move the genome into the window"
            _, m, new = rng.choice(sorted(moves)[:8])
            genome[m] = new
        raise AssertionError("repair did not reach the window in 10,000 steps")

    def build(self, genome: Genome, path: Path) -> Path:
        """The genome's GGUF: each matrix's bytes copied from the type copy of its type; metadata, norms and biases
        from the original. No quantizing."""
        original = gguf.GGUFReader(self.model.original)
        copies = {t: {x.name: x for x in gguf.GGUFReader(self.model.copy(t)).tensors} for t in set(genome.values())}
        writer = gguf.GGUFWriter(path, original.fields["general.architecture"].contents())
        for f in original.fields.values():
            if not f.name.startswith("GGUF.") and f.name not in ("general.architecture", "general.file_type"):
                writer.add_key_value(f.name, f.contents(), f.types[0], f.types[-1] if len(f.types) > 1 else None)
        writer.add_file_type(gguf.LlamaFileType.GUESSED)  # a mix of types
        tensors = [copies[genome[t.name]][t.name] if t.name in genome else t for t in original.tensors]
        for t in tensors:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for t in tensors:
            writer.write_tensor_data(t.data)
        writer.close()
        return path
