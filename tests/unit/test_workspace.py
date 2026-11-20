"""Smoke tests: every workspace package imports and documents itself."""

from __future__ import annotations

import importlib

import pytest

PACKAGES = [
    "panther_core",
    "panther_data",
    "panther_measure",
    "panther_quant",
    "panther_search",
    "panther_pipeline",
    "panther_ops",
    "panther_dashboard",
    "panther_cli",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a module docstring"
