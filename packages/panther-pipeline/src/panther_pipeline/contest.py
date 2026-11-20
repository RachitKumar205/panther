"""One contest: the resumable steps that produce a winner.

The genetic search is cached under a digest of the genome, the search parameters and the KL weight;
the winner is derived from that. This is the vertical slice: everything runs end to end on fakes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from panther_core.cache import key
from panther_core.genome import Assignment, Genome
from panther_pipeline.steps import run_step
from panther_search.ga import Evaluation, SearchParams, search

Correctness = dict[str, list[bool]]


class Contest(BaseModel):
    """One model + one RAM budget."""

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "1.5b@1"
    budget_gib: int = Field(gt=0)


class Winner(BaseModel):
    """The winning genome and its score."""

    model_config = ConfigDict(frozen=True)

    winner: dict[str, str]  # matrix -> type name
    score: float


def genome_key(genome: Genome) -> str:
    """A complete digest of everything that determines a genome's search (bug B1)."""
    return key(
        genome.geometry.model_dump(mode="json"),
        genome.ram_model.model_dump(mode="json"),
        genome.allowed,
        genome.window.model_dump(mode="json"),
        genome.error,
    )


def run_contest(
    contest: Contest,
    data_dir: Path,
    genome: Genome,
    scorer: Callable[[Assignment], Evaluation],
    original: Correctness,
    kl_weight: float,
    params: SearchParams,
) -> Winner:
    """Run the search (if needed) and derive the winner, both resumable."""
    rewrite = data_dir / "rewrite"
    ga_inputs = key("ga", contest.name, genome_key(genome), params.model_dump(mode="json"), kl_weight)
    ga_path = rewrite / f"ga-{contest.name}.json"

    finalists: list[Evaluation] = []

    def compute_ga() -> dict[str, object]:
        nonlocal finalists
        finalists = search(genome, scorer, original, kl_weight, params)
        return {"finalists": [e.model_dump(mode="json") for e in finalists]}

    run_step("ga", ga_path, ga_inputs, compute_ga)
    if not finalists:  # skipped: reuse the recorded finalists
        finalists = [Evaluation.model_validate(f) for f in json.loads(ga_path.read_text())["finalists"]]

    best = min(finalists, key=lambda e: e.score.value)
    winner = {name: str(quant) for name, quant in best.assignment.items()}
    run_step(
        "winner",
        rewrite / f"winner-{contest.name}.json",
        key("winner", ga_inputs, best.assignment),
        lambda: {"winner": winner, "score": best.score.value},
    )
    return Winner(winner=winner, score=best.score.value)
