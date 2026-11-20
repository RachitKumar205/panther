# 0001 — Record architecture decisions

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M0

## Context

The rewrite touches scientific choices (splits, fitness, search, verdict) and operational ones
(containers, claims, publishing). Undocumented decisions become tribal knowledge, and bug fixes can
silently change results.

## Decision

Every decision that constrains implementation is written as an ADR in `docs/decisions/` **before**
the phase that depends on it. ADRs are immutable; a changed decision is a new ADR that supersedes
the old one.

## Alternatives considered

- A single `DECISIONS.md` (rejected: merges unrelated decisions and hinders linking).
- Inline comments only (rejected: discoverability and review).

## Consequences

Reviewers can see *why* code looks the way it does; parity differences have a paper trail. The cost
is one short document per decision.

## Links

- `docs/rewrite-plan.md` §4, §12
