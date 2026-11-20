"""The CLI parses contests and runs its commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from panther_cli.main import app, parse_contest


@pytest.mark.parametrize(
    ("spec", "size", "budget"),
    [("14b@5", "14b", 5), ("1.5b@1", "1.5b", 1), ("22B@7", "22b", 7)],
)
def test_parse_contest(spec: str, size: str, budget: int) -> None:
    assert parse_contest(spec) == (size, budget)


@pytest.mark.parametrize("spec", ["14b", "@5", "14b@", "x@5", ""])
def test_parse_contest_rejects_bad_formats(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_contest(spec)


def test_run_parses_and_echoes_the_contest() -> None:
    result = CliRunner().invoke(app, ["run", "14b@5"])
    assert result.exit_code == 0
    assert "14b@5" in result.stdout


def test_run_rejects_a_bad_contest() -> None:
    result = CliRunner().invoke(app, ["run", "nonsense"])
    assert result.exit_code != 0
