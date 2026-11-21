"""The genetic search for one contest: the best 3 genomes that fit the budget, by fitness on the GA set."""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from panther.fitness import Fitness
from panther.model import TYPES, Model
from panther.model.model import DATA
from panther.presets import PRESETS, Budget, Presets
from panther.search.genes import Genes, Genome

POPULATION = 16  # the best 16 scored genomes are the parents
BATCH = 8  # candidates scored at once, each round
ELITE, POLISH = 4, 0.3  # 30 % of children are a mutation of one of the 4 best
SOLVED = 4  # starting candidates from CP-SAT: the lowest summed output error inside the window
PATIENCE, AT_LEAST, AT_MOST = 15, 30, 70  # stop after 15 candidates without a new best; 30 to 70 candidates
FINALISTS = 3
SCRATCH = Path(os.environ.get("PANTHER_SCRATCH", "/dev/shm"))  # candidate files live in RAM while scored


class Search:
    """Starts from the layouts of the 22 presets and of the 13 types, each repaired into the window, and 4 genomes
    from CP-SAT. Then, in rounds of 8 candidates scored at once: 7 children (30 % of the time a mutation of one of the
    4 best; otherwise a two-point slice of two parents, each the better of two random members of the population, half
    of those also mutated; then repaired into the window) and 1 from CP-SAT with each role's weight fitted to the
    scores so far. Every round is recorded in
    data/contests/<model>@<budget>/search.json, so a stopped search continues where it stopped. Every random choice
    is seeded: the same run gives the same result."""

    def __init__(self, model: Model, budget: Budget, fitness: Fitness, preset: Path, seed: int = 0) -> None:
        """fitness: on the GA set's text and its 900 search questions. preset: the competitor's file (part 2)."""
        self.model, self.fitness, self.preset, self.seed = model, fitness, preset, seed
        self.genes = Genes(model, budget)
        self.dir = DATA / "contests" / f"{model.name}@{budget.gib:g}"
        self.record = self.dir / "search.json"
        self.scratch = SCRATCH / f"panther-{model.name}@{budget.gib:g}"
        fitness.resources.reserve(BATCH * int(budget.gib * 2**30))  # a round's candidate files, in RAM

    def run(self) -> list[dict]:
        """The best 3 genomes, each with its score and its score relative to the preset's (same questions and text)."""
        state = json.loads(self.record.read_text()) if self.record.exists() else {}
        if "finalists" in state:
            return state["finalists"]
        if "preset" not in state:
            state = {"preset": asdict(self.fitness.score(self.preset)), "scored": []}
            self.save(state)
        seen = {self.key(s["genome"]) for s in state["scored"]}

        rng = random.Random(self.seed)
        presets = Presets(self.model)
        starts = [self.genes.like(presets.layout(p)) for p in PRESETS]
        starts += [self.genes.like(dict.fromkeys(self.genes.order, t)) for t in TYPES]
        starts = [self.genes.repair(g, rng) for g in starts] + self.genes.solve(SOLVED)
        unique = {self.key(g): g for g in starts}
        todo = [g for k, g in unique.items() if k not in seen]
        for i in range(0, len(todo), BATCH):
            self.evaluate(todo[i:i + BATCH], state, seen)

        while not self.done(state["scored"]):
            rng = random.Random(f"{self.seed}-{len(state['scored'])}")  # the same choices after a restart
            population = [s["genome"] for s in sorted(state["scored"], key=lambda s: s["score"]["score"])[:POPULATION]]
            size = min(BATCH, AT_MOST - len(state["scored"]))
            batch: list[Genome] = []
            while len(batch) < size - 1:  # every candidate but one: children
                child = self.child(population, rng, seen | {self.key(g) for g in batch})
                if child is None:  # no unscored child found
                    break
                batch.append(child)
            batch += self.proposal(state["scored"], batch)[: size - len(batch)]
            if len(batch) < size and (child := self.child(population, rng, seen | {self.key(g) for g in batch})):
                batch.append(child)  # the solver found nothing: one more child
            if not batch:  # nothing new to try
                break
            self.evaluate(batch, state, seen)

        best = sorted(state["scored"], key=lambda s: s["score"]["score"])[:FINALISTS]
        state["finalists"] = [{**s, "relative": 100 * s["score"]["score"] / state["preset"]["score"]} for s in best]
        self.save(state)
        return state["finalists"]

    def proposal(self, scored: list[dict], batch: list[Genome]) -> list[Genome]:
        """The solver's candidate for this round: the lowest summed output error with each role's weight fitted to
        every score so far, at least 3 matrices from everything scored and from the rest of the round. [] if none."""
        weights = self.genes.fit([s["genome"] for s in scored], [s["score"]["score"] for s in scored])
        return self.genes.solve(1, weights, away=[s["genome"] for s in scored] + batch)

    def child(self, population: list[Genome], rng: random.Random, taken: set) -> Genome | None:
        """A new child, not in taken (None if 1,000 tries found none). population: sorted best first."""
        def parent() -> Genome:
            a, b = rng.sample(range(len(population)), 2)
            return population[min(a, b)]  # the lower index is the better of the two
        for _ in range(1000):
            if rng.random() < POLISH:
                c = self.genes.mutate(rng.choice(population[:ELITE]), rng)
            else:
                c = self.genes.slice(parent(), parent(), rng)
                c = self.genes.mutate(c, rng) if rng.random() < 0.5 else c
            c = self.genes.repair(c, rng)
            if self.key(c) not in taken:
                return c
        return None

    def evaluate(self, batch: list[Genome], state: dict, seen: set) -> None:
        """Scores a batch at once (each: its file built in RAM, scored, deleted) and records it, in batch order."""
        def score(i: int, genome: Genome) -> dict:
            self.scratch.mkdir(parents=True, exist_ok=True)
            file = self.scratch / f"candidate-{i}.gguf"
            try:
                return asdict(self.fitness.score(self.genes.build(genome, file)))
            finally:
                file.unlink(missing_ok=True)  # also when building or scoring failed: it is RAM
        with ThreadPoolExecutor(len(batch)) as pool:
            scores = list(pool.map(score, range(len(batch)), batch))
        for genome, s in zip(batch, scores, strict=True):
            state["scored"].append({"genome": genome, "score": s})
            seen.add(self.key(genome))
        self.save(state)

    def done(self, scored: list[dict]) -> bool:
        scores = [s["score"]["score"] for s in scored]
        since_best = len(scores) - 1 - scores.index(min(scores))
        return len(scores) >= AT_MOST or (len(scores) >= AT_LEAST and since_best >= PATIENCE)

    def key(self, genome: Genome) -> tuple[str, ...]:
        return tuple(genome[m] for m in self.genes.order)

    def save(self, state: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record.with_suffix(".part").write_text(json.dumps(state))
        os.replace(self.record.with_suffix(".part"), self.record)
