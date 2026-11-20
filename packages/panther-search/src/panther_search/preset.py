"""llama.cpp presets and their nominal bits per weight (bug B15 fixed)."""

from __future__ import annotations

from collections.abc import Mapping

PRESETS: dict[str, float] = {
    "IQ1_S": 1.56,
    "IQ1_M": 1.75,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.5,
    "IQ2_M": 2.7,
    "Q2_K_S": 2.9,
    "IQ3_XXS": 3.06,
    "Q2_K": 2.625,  # the real bits per weight; the old table had 3.2 (bug B15)
    "IQ3_XS": 3.3,
    "IQ3_S": 3.44,
    "Q3_K_S": 3.5,
    "IQ3_M": 3.66,
    "Q3_K_M": 3.9,
    "IQ4_XS": 4.25,
    "Q3_K_L": 4.27,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.89,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.5,
}
"""Nominal bits per weight of each llama.cpp preset, high to low when iterated."""


def presets_that_fit(room_bpw: float, tolerance: float = 0.3) -> list[str]:
    """Presets not clearly above the budget, so they are worth making and measuring."""
    return [
        p
        for p, bpw in sorted(PRESETS.items(), key=lambda kv: -kv[1])
        if bpw <= room_bpw + tolerance or p == "IQ1_S"
    ]


def choose_preset(scored: Mapping[str, tuple[float, float]]) -> str:
    """Most accurate preset, ties broken by lower KL."""
    return max(scored, key=lambda p: (round(scored[p][0], 1), -scored[p][1]))
