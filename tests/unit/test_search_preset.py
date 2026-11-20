"""Preset selection uses Q2_K's real bits per weight (bug B15)."""

from __future__ import annotations

from panther_search.preset import choose_preset, presets_that_fit


def test_q2k_is_tried_when_it_can_fit() -> None:
    assert "Q2_K" in presets_that_fit(2.5)  # 2.625 <= 2.8, so it must be made
    assert "Q2_K" not in presets_that_fit(2.2)  # 2.625 > 2.5


def test_choose_preset_prefers_accuracy_then_lower_kl() -> None:
    assert choose_preset({"a": (60.0, 0.5), "b": (61.0, 0.9)}) == "b"
    assert choose_preset({"a": (60.0, 0.5), "b": (60.0, 0.3)}) == "b"
