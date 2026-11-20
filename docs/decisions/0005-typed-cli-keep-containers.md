# 0005 — Typed Python CLI, container-per-stage topology retained

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M6

## Context

The old `./panther` is a 151-line bash script that syncs code, claims resources, runs three
containers, pauses/resumes, and guards GitLab. It is hard to test and duplicates logic found in
Python (`sizes`, `claim`, `live`).

## Decision

Replace the bash orchestrator with a typed Python CLI (Typer), keeping the **same container
topology**: `fetch` (CPU/network) → `run` (GPU/no network) → `publish` (CPU/network), rootless
podman, the same mounts and the same lock/claim semantics. Shell remains only where it is the right
tool (container entrypoints, host setup, nginx).

## Alternatives considered

- **Keep bash, port only Python.** Rejected: leaves the orchestration untestable.
- **Drop containers for a process runner.** Rejected: loses GPU/network isolation and the
  reproducibility guarantees.

## Consequences

The CLI is testable with a `ContainerRuntime` fake; the container contract stays familiar to
operators. The Python CLI must reproduce the bash lock-handoff carefully (tested in M6).

## Links

- `docs/rewrite-plan.md` §5, §7 (M6)
- Bug B9, B13, B16
