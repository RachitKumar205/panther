"""The final step: pick the winner and judge it against the preset (ADR 0022)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from panther_pipeline.verdict import Correctness, Measurement, score, versus


class Finalist(BaseModel):
    """A candidate file: its assignment and measurement."""

    model_config = ConfigDict(frozen=True)

    assignment: dict[str, str]  # matrix -> type name
    measurement: Measurement  # correctness + KL chunks


class FinalResult(BaseModel):
    """The winner and its verdict against the preset."""

    model_config = ConfigDict(frozen=True)

    assignment: dict[str, str]  # matrix -> type name
    score: float  # flips + 10 * KL
    relative: float  # 100 * ours / preset
    ci: tuple[float, float]  # 95% interval of the relative score
    verdict: str  # WIN / TIE / LOSE


def choose_winner(finalists: list[Finalist], original: Correctness, kl_weight: float = 10.0) -> Finalist:
    """The finalist with the lowest score."""
    return min(finalists, key=lambda f: score(f.measurement, original, kl_weight))


def final(
    finalists: list[Finalist],
    preset: Measurement,
    original: Correctness,
    kl_weight: float = 10.0,
) -> FinalResult:
    """Pick the winner and return its verdict against the preset."""
    winner = choose_winner(finalists, original, kl_weight)
    result = versus(winner.measurement, preset, original, kl_weight)
    return FinalResult(
        assignment=winner.assignment,
        score=score(winner.measurement, original, kl_weight),
        relative=result.relative,
        ci=result.ci,
        verdict=result.verdict,
    )
