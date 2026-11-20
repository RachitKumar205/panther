"""The final step picks the lowest score and judges it; the bundle manifest is fixed."""

from __future__ import annotations

from panther_pipeline.bundle import BundleManifest
from panther_pipeline.final import Finalist, final
from panther_pipeline.verdict import Measurement


def make(correct: list[bool], kl_chunks: tuple[float, ...]) -> Measurement:
    return Measurement(correct={"mmlu": correct}, kl_chunks=kl_chunks)


ORIGINAL = {"mmlu": [True] * 100}


def test_choose_winner_picks_the_lowest_score() -> None:
    good = Finalist(assignment={"a": "Q4_K"}, measurement=make([True] * 100, (0.1, 0.1)))
    bad = Finalist(assignment={"a": "Q2_K"}, measurement=make([True] * 50 + [False] * 50, (0.5, 0.5)))
    result = final([good, bad], preset=make([True] * 50 + [False] * 50, (0.5, 0.5)), original=ORIGINAL)
    assert result.assignment == {"a": "Q4_K"}
    assert result.verdict == "WIN"


def test_bundle_manifest_is_frozen_and_complete() -> None:
    manifest = BundleManifest(
        file="m.gguf",
        sha256="0" * 64,
        budget_gib=5,
        predicted_ram_gib=4.7,
        context=4096,
        kv_cache_type="Q8_0",
    )
    assert manifest.budget_gib == 5
    assert manifest.kv_cache_type == "Q8_0"
