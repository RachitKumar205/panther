"""Genome operators preserve the RAM window and are deterministic."""

from __future__ import annotations

import random

from panther_core.genome import Genome, RamWindow
from panther_core.geometry import GeometrySpec, ModelGeometry
from panther_core.quant import IMATRIX_FORMATS, SEARCH_FORMATS, QuantType
from panther_core.ram import RamModel, RuntimeConfig


def build() -> tuple[Genome, RamModel, ModelGeometry]:
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
    ram_model = RamModel(geometry=geometry, runtime=RuntimeConfig(4096, "Q8_0", 0.4))
    allowed = geometry.allowed(SEARCH_FORMATS, IMATRIX_FORMATS)
    # Higher bits -> lower error, so repair can rank moves.
    error = {n: {t: 10.0 / t.bits_per_weight for t in ts} for n, ts in allowed.items()}
    window = RamWindow(
        ram_model.total_bytes({n: QuantType.Q2_K for n in allowed}),
        ram_model.total_bytes({n: QuantType.Q4_K for n in allowed}),
    )
    genome = Genome(geometry, ram_model, allowed, window, error, rng=random.Random(0))
    return genome, ram_model, geometry


def test_repair_brings_an_oversized_genome_into_the_window() -> None:
    genome, _, _ = build()
    repaired = genome.repair(genome.uniform(QuantType.Q8_0))
    assert genome.fits(repaired)


def test_repair_keeps_an_in_window_genome() -> None:
    genome, _, _ = build()
    assignment = genome.uniform(QuantType.Q3_K)
    assert genome.fits(assignment)
    assert genome.repair(assignment) == assignment


def test_mutate_and_crossover_stay_in_the_window() -> None:
    genome, _, _ = build()
    seed = genome.uniform(QuantType.Q4_K)
    for _ in range(50):
        assert genome.fits(genome.mutate(seed))
        assert genome.fits(genome.crossover(seed, genome.uniform(QuantType.Q2_K)))


def test_same_seed_gives_the_same_mutants() -> None:
    def mutate() -> list[QuantType]:
        genome, _, _ = build()
        out = genome.uniform(QuantType.Q4_K)
        for _ in range(5):
            out = genome.mutate(out)
        return [out[name] for name in sorted(out)]

    assert mutate() == mutate()
