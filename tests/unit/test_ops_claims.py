"""Claims reserve against the caps and refuse what does not fit."""

from __future__ import annotations

import pytest

from panther_ops.claims import Claim, Claims, Refused


def test_claims_reserve_and_refuse_against_the_caps() -> None:
    claims = Claims(gpu_cap=100, ram_cap=200)
    claims.reserve(Claim(id="a", gpu=40, ram=80))
    claims.reserve(Claim(id="b", gpu=40, ram=80))
    assert set(claims.active) == {"a", "b"}
    with pytest.raises(Refused):
        claims.reserve(Claim(id="c", gpu=30, ram=80))  # GPU: 40+40+30 > 100


def test_dropping_a_claim_frees_room() -> None:
    claims = Claims(gpu_cap=100, ram_cap=200)
    claims.reserve(Claim(id="a", gpu=60, ram=100))
    claims.drop("a")
    claims.reserve(Claim(id="b", gpu=60, ram=100))  # now fits
    assert set(claims.active) == {"b"}


def test_fits_is_additive_over_both_resources() -> None:
    claims = Claims(gpu_cap=10, ram_cap=10)
    claims.reserve(Claim(id="a", gpu=5, ram=1))
    assert claims.fits(4, 20) is False  # RAM would overflow
    assert claims.fits(6, 1) is False  # GPU would overflow
