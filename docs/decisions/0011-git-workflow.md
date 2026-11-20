# 0011 — Git workflow: all work on `lalit`

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M0

## Context

The repository has an old `master` and an unrelated historical `rewrite-plan` branch. We need one
place where the rewrite lands, with reviewable history.

## Decision

- Remote: `origin = https://github.com/RachitKumar205/panther.git`.
- All work on the `lalit` branch; `master` is never committed to.
- Conventional Commits; one phase may be several green commits.
- `gh` is used for repository operations; `git push origin lalit` for code.
- Commits and pushes happen frequently (after each coherent change), without asking.
- Optional PRs target `master` and summarize the phase and ledger items closed.
- `v1.0.0` tag at cutover.

## Alternatives considered

- **Work on `master`.** Rejected: loses the ability to compare and review the rewrite.
- **Use the historical `rewrite-plan` branch.** Rejected: it is an old, unrelated implementation.

## Consequences

`lalit` diverges completely from `master` (the old tree is removed there); the old code remains on
`master` and in the local `panther-old/` reference.

## Links

- `docs/rewrite-plan.md` §11
