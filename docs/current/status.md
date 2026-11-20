# Status

**Last updated:** 2026-11-20
**Current phase:** M0 — Bootstrap (in progress)

| Phase | Status | Notes |
|---|---|---|
| M0 Bootstrap | In progress | plan, ADRs, git branch, ignore rules done; workspace/tooling next |
| M1 `panther-core` | Planned | — |
| M2 `panther-data` | Planned | — |
| M3 `panther-measure` | Planned | — |
| M4 Vertical slice | Planned | — |
| M5 `panther-quant` | Planned | — |
| M6 `panther-ops` + CLI | Planned | — |
| M7 Dashboard | Planned | — |
| M8 Pi verifier | Planned | — |
| M9 Deploy | Planned | — |
| M10 Parity + E2E | Planned | — |
| M11 Cutover | Planned | — |

## Open work in the current phase

- Create the uv workspace and the eight package members plus `apps/panther-cli`.
- Add ruff, pyright, pytest, hypothesis, pytest-cov, import-linter, pydantic.
- Add the CI workflow.
- Capture reference fixtures from `panther-old` into `tests/parity/fixtures/`.

## Recently completed

- `lalit` branch created from `master`; old tree removed from the branch (kept on `master` and in
  `panther-old/`).
- `.gitignore` and `.containerignore` rewritten; `panther-old/` ignored (B20).
- Rewrite plan written; ADRs 0001–0011 recorded.
