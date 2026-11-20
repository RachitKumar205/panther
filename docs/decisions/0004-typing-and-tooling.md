# 0004 — Typing and tooling: pyright strict, ruff, pydantic v2

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M0

## Context

The old code used loose dicts, bare tuples and `Any`-heavy boundaries (`how: dict`, `r: dict`,
`a: dict[str, str]`). Bugs such as incomplete cache keys and mixed units are hard to catch.

## Decision

- **pyright** in `strict` mode for every package.
- **ruff** for lint and format.
- **pydantic v2** for every persisted or external data boundary; plain dataclasses only for
  internal immutable values.
- Docstrings on all public modules, classes and functions.
- `from __future__ import annotations` everywhere.

## Alternatives considered

- **mypy strict.** Rejected: pyright was already the project's choice and has better editor
  integration here.
- **pyright basic.** Rejected: too weak to catch the boundary bugs we are fixing.

## Consequences

Strict typing slows initial development slightly but makes the cache/record contracts explicit and
catches the exact classes of bugs in the ledger.

## Links

- `docs/rewrite-plan.md` §10
