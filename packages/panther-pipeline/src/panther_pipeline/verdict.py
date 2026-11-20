"""The verdict: relative score and its interval against the preset (ADR 0022)."""

from __future__ import annotations

from math import hypot

from pydantic import BaseModel, ConfigDict

from panther_core.stats import flips, kl_diff_std_error, kl_mean, paired_flip_std_error, verdict

Correctness = dict[str, list[bool]]


class Measurement(BaseModel):
    """A file's measurement: per-task correctness and per-chunk KL."""

    model_config = ConfigDict(frozen=True)

    correct: Correctness  # task -> per-question right/wrong
    kl_chunks: tuple[float, ...]  # per-chunk KL means


class VerdictResult(BaseModel):
    """A relative score, its 95% interval and the verdict."""

    model_config = ConfigDict(frozen=True)

    relative: float  # 100 * ours / preset
    ci: tuple[float, float]  # 95% interval of the relative score
    verdict: str  # WIN / TIE / LOSE


def score(measurement: Measurement, original: Correctness, kl_weight: float = 10.0) -> float:
    """``flips% + kl_weight * KL`` (lower is better)."""
    return flips(measurement.correct, original) + kl_weight * kl_mean(measurement.kl_chunks)


def versus(
    ours: Measurement,
    preset: Measurement,
    original: Correctness,
    kl_weight: float = 10.0,
    z: float = 1.96,
) -> VerdictResult:
    """Our relative score vs the preset, with its 95% interval and WIN/TIE/LOSE verdict."""
    ours_score = score(ours, original, kl_weight)
    preset_score = score(preset, original, kl_weight)
    flips_se = paired_flip_std_error(ours.correct, preset.correct, original)
    kl_se = kl_diff_std_error(ours.kl_chunks, preset.kl_chunks)
    se = hypot(flips_se, kl_weight * kl_se)
    lo, hi = 100 * (ours_score - z * se) / preset_score, 100 * (ours_score + z * se) / preset_score
    return VerdictResult(
        relative=100 * ours_score / preset_score,
        ci=(lo, hi),
        verdict=verdict(lo, hi).value,
    )
