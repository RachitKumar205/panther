"""The genetic search is deterministic (B8) and its improvement SE is paired (B6)."""

from __future__ import annotations

import random

from panther_core.genome import Assignment, Genome, RamWindow
from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType
from panther_core.ram import RamModel, RuntimeConfig
from panther_core.stats import Score
from panther_search.ga import Evaluation, SearchParams, improvement, search

ORIGINAL = {"mmlu": [True] * 200}


def build_genome(seed: int) -> Genome:
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
    return Genome(geometry, ram_model, allowed, window, error, rng=random.Random(seed))


def scorer(assignment: Assignment) -> Evaluation:
    bpw = sum(q.bits_per_weight for q in assignment.values()) / len(assignment)
    return Evaluation(
        assignment=assignment,
        score=Score(flips=bpw * 10, kl=0.0, kl_weight=10.0, flips_se=0.5, kl_se=0.0),
        correct={"mmlu": [True] * 200},
        kl_chunks=(0.0, 0.0, 0.0),
    )


def test_search_is_deterministic_given_a_seed() -> None:
    params = SearchParams(seed=0, min_evals=30, max_evals=50)
    a = search(build_genome(0), scorer, ORIGINAL, 10.0, params)
    b = search(build_genome(0), scorer, ORIGINAL, 10.0, params)
    assert [e.assignment for e in a] == [e.assignment for e in b]


def test_search_returns_the_top_finalists_sorted() -> None:
    result = search(build_genome(1), scorer, ORIGINAL, 10.0, SearchParams(seed=1, min_evals=30, max_evals=50, top=3))
    assert len(result) == 3
    assert [e.score.value for e in result] == sorted(e.score.value for e in result)


def test_improvement_is_zero_for_identical_genomes() -> None:
    e = Evaluation(
        assignment={},
        score=Score(flips=0.0, kl=0.0),
        correct={"mmlu": [True, False] * 100},
        kl_chunks=(1.0, 2.0, 3.0),
    )
    assert improvement(e, e, ORIGINAL, 10.0) == 0.0
