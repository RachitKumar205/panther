# 0002 — Behavioural parity, except intentional bug fixes

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** all

## Context

The previous pipeline produced results that are compared in the project README. A rewrite that
quietly changes the science would invalidate that comparison; a rewrite that reproduces every bug
would defeat the purpose.

## Decision

v2 reproduces the old algorithms and outputs on identical inputs, **except** where a bug in the
audit (B1–B21) is fixed. Every intentional difference is recorded in the bug ledger and pinned by a
regression test. The parity harness (M10) diffs old and new records and must show either zero
difference or a documented one.

## Alternatives considered

- **Free to improve.** Rejected: too easy to change the science unnoticed.
- **Exact output parity.** Rejected: it would preserve known defects.

## Consequences

Parity is a hard gate, so fixtures and the fake backends must be faithful. Bug fixes must state
their expected effect on results.

## Links

- `docs/rewrite-plan.md` §3, §8, §9
- Bug ledger B1–B21
