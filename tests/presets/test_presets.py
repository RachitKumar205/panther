"""The RAM model, the presets' sizes and files, and the choice of competitor."""

from __future__ import annotations

import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from panther.model import Model
from panther.model import model as model_module
from panther.presets import PRESETS, Budget, Presets
from panther.presets import presets as presets_module
from panther.presets.budget import GIB
from tests.model.tiny import HEADS, HIDDEN, KV_HEADS, LAYERS, needs_llama, write_imatrix, write_original

MIB = 2**20


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Model:
    """The tiny model (part 1's test model), with its original and importance matrix."""
    monkeypatch.setattr(model_module, "DATA", tmp_path)
    monkeypatch.setattr(presets_module, "DATA", tmp_path)
    m = Model.load("1.5b")
    m.dir.mkdir(parents=True)
    write_original(m.original)
    write_imatrix(m.original, m.imatrix)
    return m


def test_twenty_two_presets() -> None:
    assert len(PRESETS) == len(set(PRESETS)) == 22
    assert not {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "TQ1_0", "TQ2_0", "MXFP4_MOE"} & set(PRESETS)


# --- the RAM model ---

def test_kv_cache_is_keys_and_values_of_every_layer_at_the_context_in_8_bits(m: Model) -> None:
    head = HIDDEN // HEADS
    assert Budget(m, 1).kv_cache == 2 * LAYERS * KV_HEADS * head * 4096 * 34 // 32


def test_kv_cache_uses_the_head_width_the_gguf_states(m: Model) -> None:
    # Mistral Small 24B: 32 heads of 128 on a width of 5120 (not 5120 / 32 = 160); its GGUF states key_length
    w = gguf.GGUFWriter(m.original, "llama")
    w.add_embedding_length(5120)
    w.add_block_count(40)
    w.add_head_count(32)
    w.add_head_count_kv(8)
    w.add_key_length(128)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()
    assert Budget(m, 7).kv_cache == 2 * 40 * 8 * 128 * 4096 * 34 // 32


def test_ram_is_file_plus_kv_cache_plus_overhead_and_fits_up_to_1_005_of_the_budget(m: Model) -> None:
    b = Budget(m, 1)
    assert b.ram(1000) == 1000 + b.kv_cache + int(0.40 * GIB)
    limit = int(1.005 * GIB) - b.kv_cache - b.overhead
    assert b.fits(limit) and not b.fits(limit + 1)


# --- sizes, files and the competitor (real llama-quantize on the tiny model) ---

@needs_llama
def test_dry_run_sizes_match_the_built_files(m: Model) -> None:
    presets = Presets(m)
    for p in ("IQ1_S", "Q2_K", "Q4_K_M", "Q8_0"):
        tensors = sum(int(t.n_bytes) for t in gguf.GGUFReader(presets.build(p)).tensors)
        assert abs(presets.size(p) - tensors) <= 0.01 * MIB  # dry-run prints MiB to 2 decimals
    assert presets.size("IQ1_S") < presets.size("Q4_K_M") < presets.size("Q8_0")


@needs_llama
def test_every_preset_name_is_one_llama_quantize_knows(m: Model) -> None:
    assert all(Presets(m).size(p) > 0 for p in PRESETS)


def budget_between(m: Model, small: str, large: str) -> Budget:
    """A budget the small preset's file fits and the large one's does not (the tiny model needs fractions of a GiB)."""
    presets = Presets(m)
    b = Budget(m, 1)
    middle = (presets.build(small).stat().st_size + presets.build(large).stat().st_size) // 2
    return Budget(m, b.ram(middle) / 1.005 / GIB)


@needs_llama
def test_the_competitor_is_the_fitting_preset_with_the_lowest_score(m: Model) -> None:
    presets = Presets(m)
    budget = budget_between(m, "Q3_K_M", "Q4_K_S")
    scored = []

    def score(file: Path) -> float:
        scored.append(file.stem)
        return {"Q2_K": 1.0, "IQ3_XS": 0.5}.get(file.stem, 2.0)  # IQ3_XS: best here, though not the largest
    result = presets.best(budget, score)
    built = [p for p in PRESETS if presets.file(p).exists()]  # only presets whose dry-run size fits were built
    fitting = [p for p in built if budget.fits(presets.file(p).stat().st_size)]
    assert result["preset"] == "IQ3_XS" and not result["unfair"]
    assert set(scored) == set(fitting) and "Q4_K_S" not in scored and "Q3_K_M" in scored
    assert "Q8_0" not in built
    assert result["scores"]["Q2_K"] == 1.0 and result["ram_gib"] <= 1.005 * budget.gib


@needs_llama
def test_the_choice_is_saved_and_reused(m: Model, tmp_path: Path) -> None:
    presets = Presets(m)
    budget = budget_between(m, "Q3_K_M", "Q4_K_S")
    first = presets.best(budget, lambda f: 1.0)
    assert json.loads((tmp_path / f"contests/{m.name}@{budget.gib:g}/preset.json").read_text()) == first
    assert presets.best(budget, lambda f: pytest.fail("scored again")) == first


@needs_llama
def test_a_whole_budget_names_one_record_and_numpy_scores_are_saved(m: Model, tmp_path: Path) -> None:
    Presets(m).best(Budget(m, 1), lambda f: np.float32(0.5))
    assert (tmp_path / f"contests/{m.name}@1/preset.json").exists()
    assert Presets(m).best(Budget(m, 1.0), lambda f: pytest.fail("scored again"))["scores"]["IQ1_S"] == 0.5


@needs_llama
def test_if_none_fits_the_smallest_competes_and_is_marked_unfair(m: Model) -> None:
    presets = Presets(m)
    budget = Budget(m, 0.001)  # below the KV cache and overhead alone
    result = presets.best(budget, lambda f: 1.0)
    assert result["preset"] == min(PRESETS, key=presets.size) and result["unfair"]
    assert result["ram_gib"] > budget.gib


@needs_llama
def test_presets_no_contest_chose_are_deleted(m: Model) -> None:
    presets = Presets(m)
    budget = budget_between(m, "Q3_K_M", "Q4_K_S")
    chosen = presets.best(budget, lambda f: 0.0 if f.stem == "Q2_K" else 1.0)["preset"]
    presets.build("Q8_0")
    presets.delete_unused()
    assert [f.stem for f in presets.dir.glob("*.gguf")] == [chosen]
