"""The `panther` command-line entry point.

The full command surface is implemented in M6; this bootstraps the Typer application so the
workspace, packaging and tests can be validated end to end.
"""

from __future__ import annotations

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


def main() -> None:
    """Run the CLI."""
    app()
