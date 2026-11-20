"""The genetic search, with bugs B6 and B8 fixed.

Steady state. Evaluations are taken in a deterministic order (B8), and a new best counts as an
improvement only if it beats the last *counted* improvement by more than the standard error of the
difference (B6 — the old rule ratcheted on every sub-SE gain).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from panther_core.genome import Assignment, Genome
from panther_core.quant import QuantType
from panther_core.stats import Score, kl_diff_std_error, paired_flip_std_error

Original = Mapping[str, Sequence[bool]]


class Evaluation(BaseModel):
    """One measured genome."""

    model_config = ConfigDict(frozen=True)

    assignment: dict[str, QuantType]  # matrix -> type
    score: Score  # value and standard errors
    correct: dict[str, list[bool]]  # task -> per-question correctness
    kl_chunks: tuple[float, ...]  # per-chunk KL means


class SearchParams(BaseModel):
    """Genetic-search hyperparameters."""

    model_config = ConfigDict(frozen=True)

    population: int = Field(default=16, ge=1)
    elite: int = Field(default=4, ge=1)
    min_evals: int = Field(default=30, ge=1)
    max_evals: int = Field(default=70, ge=1)
    patience: int = Field(default=15, ge=1)
    top: int = Field(default=3, ge=1)
    seed: int = 0


def improvement(prev: Evaluation, new: Evaluation, original: Original, kl_weight: float) -> float:
    """Standard error of the score difference between two genomes (paired over questions/chunks)."""
    flips_se = paired_flip_std_error(new.correct, prev.correct, original)
    kl_se = kl_diff_std_error(new.kl_chunks, prev.kl_chunks)
    return float(np.hypot(flips_se, kl_weight * kl_se))


def search(
    genome: Genome,
    scorer: Callable[[Assignment], Evaluation],
    original: Original,
    kl_weight: float,
    params: SearchParams,
    preset: Assignment | None = None,
) -> list[Evaluation]:
    """Run the steady-state genetic search and return the best ``params.top`` evaluations."""
    population = sorted((scorer(a) for a in _initial(genome, preset, params.population)), key=_by_score)
    population = population[: params.population]
    best = reference = population[0]
    last_gain = len(population)
    n = len(population)

    for _ in range(params.max_evals - n):
        candidate = scorer(_child(genome, population, params.elite))
        n += 1
        if candidate.score.value < best.score.value:
            best = candidate
            se = improvement(reference, candidate, original, kl_weight)
            if reference.score.value - candidate.score.value > se:
                reference, last_gain = candidate, n
        population = sorted([*population, candidate], key=_by_score)[: params.population]
        if n >= params.min_evals and n - last_gain >= params.patience:
            break
    return sorted(population, key=_by_score)[: params.top]


def _by_score(evaluation: Evaluation) -> float:
    return evaluation.score.value


def _initial(genome: Genome, preset: Assignment | None, size: int) -> list[Assignment]:
    all_types = {t for ts in genome.allowed.values() for t in ts}
    universal = sorted(
        (t for t in all_types if all(t in genome.allowed[n] for n in genome.allowed)),
        key=lambda q: q.bits_per_weight,
    )
    pop: list[Assignment] = [dict(preset)] if preset else []
    pop += [genome.uniform(t) for t in universal if genome.fits(genome.uniform(t))]
    while len(pop) < size and pop:
        mutant = genome.mutate(genome.rng.choice(pop))
        if genome.fits(mutant):
            pop.append(mutant)
    return pop


def _child(genome: Genome, population: list[Evaluation], elite: int) -> Assignment:
    if genome.rng.random() < 0.3:
        return genome.mutate(genome.rng.choice(population[:elite]).assignment)
    p = _pick(genome, population)
    q = _pick(genome, population)
    child = genome.crossover(p, q)
    return genome.mutate(child) if genome.rng.random() < 0.5 else child


def _pick(genome: Genome, population: list[Evaluation]) -> Assignment:
    a, b = genome.rng.sample(population, 2)
    return (a if a.score.value <= b.score.value else b).assignment
