"""The full fake slice: search -> final -> bundle, end to end on fakes."""

from __future__ import annotations

import random

from panther_core.genome import Assignment, Genome, RamWindow
from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType
from panther_core.ram import RamModel, RuntimeConfig
from panther_core.stats import Score
from panther_pipeline.bundle import BundleManifest
from panther_pipeline.final import Finalist, final
from panther_pipeline.verdict import Measurement
from panther_search.ga import Evaluation, SearchParams, search

ORIGINAL = {"mmlu": [True] * 200}


def build_genome() -> Genome:
    spec = GeometrySpec(
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        vocab_size=320,
        head_dim=32,
        tie_word_embeddings=True,
        model_type="qwen2",
    )
    geometry = ModelGeometry.from_spec(spec)
    ram_model = RamModel(
        geometry=geometry,
        runtime=RuntimeConfig(context_tokens=4096, kv_cache_type="Q8_0", overhead_gib=0.4),
    )
    allowed = geometry.allowed(SEARCH_FORMATS, IMATRIX_FORMATS)
    error = {n: {t: 10.0 / t.bits_per_weight for t in ts} for n, ts in allowed.items()}
    window = RamWindow(
        lo=ram_model.total_bytes({n: QuantType.Q2_K for n in allowed}),
        hi=ram_model.total_bytes({n: QuantType.Q4_K for n in allowed}),
    )
    return Genome(geometry, ram_model, allowed, window, error, rng=random.Random(0))


def scorer(assignment: Assignment) -> Evaluation:
    bpw = sum(q.bits_per_weight for q in assignment.values()) / len(assignment)
    return Evaluation(
        assignment=assignment,
        score=Score(flips=bpw * 10, kl=0.0, kl_weight=10.0, flips_se=0.5, kl_se=0.0),
        correct={"mmlu": [True] * 200},
        kl_chunks=(0.0, 0.0, 0.0),
    )


def test_full_slice_produces_winner_verdict_and_bundle() -> None:
    params = SearchParams(seed=0, min_evals=30, max_evals=50, top=3)
    finalists = search(build_genome(), scorer, ORIGINAL, 10.0, params)
    candidates = [
        Finalist(
            assignment={name: str(quant) for name, quant in e.assignment.items()},
            measurement=Measurement(correct=e.correct, kl_chunks=e.kl_chunks),
        )
        for e in finalists
    ]
    preset = Measurement(correct={"mmlu": [True] * 100 + [False] * 100}, kl_chunks=(0.5, 0.5, 0.5))
    result = final(candidates, preset, ORIGINAL, 10.0)
    assert result.verdict in ("WIN", "TIE", "LOSE")
    manifest = BundleManifest(
        file="winner.gguf",
        sha256="0" * 64,
        budget_gib=1,
        predicted_ram_gib=0.9,
        context=4096,
        kv_cache_type="Q8_0",
    )
    assert manifest.file == "winner.gguf"
