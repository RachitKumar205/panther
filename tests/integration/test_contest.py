"""The contest runs end to end and resumes until its inputs change (B1)."""

from __future__ import annotations

import json
import random

from panther_core.genome import Assignment, Genome, RamWindow
from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType
from panther_core.ram import RamModel, RuntimeConfig
from panther_core.stats import Score
from panther_pipeline.contest import Contest, run_contest
from panther_search.ga import Evaluation, SearchParams

ORIGINAL = {"mmlu": [True] * 200}


def build() -> Genome:
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


def test_contest_is_resumable_and_invalidates_on_input_change(tmp_path) -> None:
    contest = Contest(name="1.5b@1", budget_gib=1)
    params = SearchParams(seed=0, min_evals=30, max_evals=50)
    ga = tmp_path / "rewrite" / "ga-1.5b@1.json"

    first = run_contest(contest, tmp_path, build(), scorer, ORIGINAL, 10.0, params)
    inputs1 = json.loads(ga.read_text())["inputs"]
    # Identical inputs: the search record is not rewritten.
    assert run_contest(contest, tmp_path, build(), scorer, ORIGINAL, 10.0, params) == first
    assert json.loads(ga.read_text())["inputs"] == inputs1
    # A changed parameter rewrites the record.
    run_contest(contest, tmp_path, build(), scorer, ORIGINAL, 10.0, SearchParams(seed=0, min_evals=30, max_evals=51))
    assert json.loads(ga.read_text())["inputs"] != inputs1
