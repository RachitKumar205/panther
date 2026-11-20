"""The `panther` command-line entry point."""

from __future__ import annotations

import re

import typer

app = typer.Typer(
    add_completion=False,
    help="Panther: mixed-precision model allocation within a RAM budget.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Panther: mixed-precision model allocation within a RAM budget."""


@app.command()
def version() -> None:
    """Print the Panther version."""
    typer.echo("panther 0.1.0")


def parse_contest(spec: str) -> tuple[str, int]:
    """Parse ``14b@5`` into ``(size, budget)``; raises ValueError on a bad format."""
    match = re.fullmatch(r"([0-9.]+[bB])@([0-9]+)", spec)
    if match is None:
        raise ValueError(f"expected SIZE@BUDGET like 14b@5, got {spec!r}")
    return match.group(1).lower(), int(match.group(2))


@app.command()
def run(spec: str) -> None:
    """Start a contest, e.g. ``panther run 14b@5`` (orchestration wired in M6)."""
    try:
        size, budget = parse_contest(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"contest {size}@{budget}")


@app.command()
def stop(spec: str) -> None:
    """Stop a contest's run (``SIZE@BUDGET``)."""
    try:
        size, budget = parse_contest(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"stopping {size}@{budget}")


@app.command()
def status() -> None:
    """Show containers, active runs and claims (wired in M6)."""
    typer.echo("no runs")


def main() -> None:
    """Run the CLI."""
    app()
