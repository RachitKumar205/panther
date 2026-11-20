"""The CLI entry point is wired into the workspace."""

from __future__ import annotations

from panther_cli.main import app
from typer.testing import CliRunner


def test_help_lists_the_command() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Panther" in result.stdout


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "panther 0.1.0" in result.stdout
