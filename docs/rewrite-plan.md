# Panther v2 — Rewrite Plan

| | |
|---|---|
| **Status** | Active |
| **Branch** | `lalit` on `github.com/RachitKumar205/panther` |
| **Reference implementation** | `panther-old/` (git-ignored; also the old `master` branch) |
| **Target Python** | 3.12 |
| **Last updated** | 2026-11-20 |

This document is the **single source of truth** for the rewrite. Every phase below is executed
one at a time; a phase is not started until its decisions are recorded in `docs/decisions/` and
not closed until its exit criteria are met and `docs/current/` is updated.

---

## 1. Purpose

Panther builds a llama.cpp GGUF file for each (model, RAM budget) **contest** — e.g. `14b@5` — that
fits a Raspberry Pi 5's memory and loses less quality than llama.cpp's best ready-made preset that
also fits. The existing implementation works but has a documented set of correctness, search and
operational defects. This rewrite reimplements the same pipeline with:

- every audited bug fixed and pinned by a regression test,
- strict typing and idiomatic structure,
- shared code extracted into reusable libraries,
- comprehensive unit and integration tests,
- a reproducible, documented development process.

The scientific method (splits, fitness, search, distillation, verdict) is **preserved**. The
implementation is rebuilt for correctness, testability and maintainability.

## 2. Goals and non-goals

### Goals

1. **Behavioural parity** with `panther-old` on identical inputs, except where a bug fix
   intentionally changes a result. Every intentional difference is recorded in the bug ledger.
2. **All audited bugs fixed** (B1–B21 below), each with a named regression test.
3. **Typed Python everywhere**: pyright `strict`, pydantic v2 models at every boundary, docstrings
   on all public APIs.
4. **Reuse, don't rebuild**: use existing, maintained Python packages for functionality that is not
   Panther-specific (see §10).
5. **Layered libraries** with an enforced one-way dependency graph, so domain logic is reusable and
   testable without a GPU, network or containers.
6. **Comprehensive tests**: fast unit tests in CI, integration tests against fakes, opt-in real
   GPU/network tiers, and a golden parity harness against the old implementation.
7. **Actionable operations**: the same container-per-stage topology, a real CLI, claims/locks,
   dashboard and Pi verification — all with the same public contracts.

### Non-goals

- Changing the model formats, the 14 llama.cpp types, or the mathematical definition of the score.
- Adding new models or budgets in this rewrite (data/config changes are separate).
- Replacing llama.cpp, torch, transformers, gguf, ortools or Hugging Face with home-grown code.
- Building a web framework, HTTP server, CLI framework, config parser or lock library from scratch.

## 3. Principles

1. **Reuse first.** If the standard library or a maintained package does it, use it. New code is
   only for Panther-specific domain logic. See the approved dependency table in §10.
2. **Parity except bugs.** When in doubt, match the old behaviour; if a bug fix changes it, write
   the difference down and test it.
3. **Boundaries are typed.** Every external dependency (llama.cpp, HF, podman, clock, filesystem)
   sits behind a `typing.Protocol` with a real and a fake implementation.
4. **One source of truth for constants.** Caps, paths, version strings and contract values live in
   one typed config object and are rendered into docs/dashboard; never duplicated as literals.
5. **Content-addressed everything.** A result is valid only if the key hashes *every* input that
   can change it (files, commit, constants, config, code version).
6. **Small, reversible commits.** Each phase is independently reviewable, testable and revertible.
7. **Docs are part of the definition of done.** A change without its `docs/current/` update and,
   where relevant, an ADR, is not done.

## 4. Locked decisions

The decisions below were agreed before implementation. Each has (or will have) an ADR in
`docs/decisions/` using the template in §12.

| ADR | Decision |
|---|---|
| 0001 | Record architecture decisions as ADRs before the phase that depends on them |
| 0002 | Parity with `panther-old` except intentional, logged bug fixes |
| 0003 | uv workspace with layered packages (`core`, `data`, `measure`, `quant`, `search`, `pipeline`, `ops`, `dashboard`, `cli`) |
| 0004 | pyright `strict`, ruff, pydantic v2; docstrings on public APIs |
| 0005 | Typed Python CLI (Typer) while keeping the container-per-stage topology |
| 0006 | Fakes + golden parity in CI; GPU/network/slow tests behind pytest markers |
| 0007 | Keep public contracts stable: HF repo naming, Pi manifest, dashboard `/api/state`, `claims.json`; version internal records |
| 0008 | Python 3.12 |
| 0009 | First milestone is a vertical slice on fixtures with fake backends |
| 0010 | Reuse existing libraries; new code only for Panther-specific logic |
| 0011 | Git workflow: all work on `lalit`, Conventional Commits, `gh` for remote operations |

## 5. Target architecture

### 5.1 Package graph

```
panther-core        no Panther deps. Pure domain: types, geometry, RAM model, genome,
                    scoring math, statistics, content-addressed cache keys.
panther-data        → core. Config schemas, split manifests, prompt building, integrity checks.
panther-measure     → core, data. LlamaBackend protocol + subprocess impl, parsers, result cache.
panther-quant       → core, data. StoreProvider/Distiller protocols + gguf/torch impl,
                    imatrix, GPTQ rounding, distillation.
panther-search      → core. Preset selection, genetic search, CP-SAT proposals. No I/O.
panther-pipeline    → core, data, measure, quant, search. Step graph, final/guard/verdict, bundle.
panther-ops         → core. ContainerRuntime protocol, claims, locks, guard, build/status.
panther-dashboard   → core, data. State assembly, masking, HTTP/SSE, static frontend.
panther-cli         → ops, pipeline. The `panther` entry point (Typer).
```

**Enforced rule:** arrows point one way only. `panther-core` must never import `torch`,
`transformers`, `huggingface_hub`, `podman`, `ortools` or an HTTP framework. `import-linter`
contracts run in CI and fail on violation.

### 5.2 Protocols (the seams that make testing possible)

| Protocol | Real implementation | Fake for tests |
|---|---|---|
| `LlamaBackend` | subprocess calls to `llama-perplexity` / `llama-server` | deterministic fixture backend |
| `StoreProvider` | GGUF/torch store build/restore | in-memory/tmp fixture store |
| `Distiller` | torch distillation | no-op / fixture distiller |
| `HuggingFaceRepo` | `huggingface_hub` | local directory fake |
| `ContainerRuntime` | rootless podman | recording fake |
| `Clock` | system UTC/IST | frozen clock |

### 5.3 Core data models (pydantic v2 unless noted)

- **Config:** `ModelConfig`, `ModelsConfig`, `SearchConfig`, `Contest`, `RuntimeConfig`.
- **Domain:** `Matrix`, `ModelGeometry`, `RamModel`, `Genome`, `Assignment`, `RamWindow`.
- **Measurement:** `MeasurementRecord`, `TaskResult`, `KlResult`, `PplResult`, `Score`,
  `RelativeScore`, `BootstrapInterval`.
- **Pipeline:** `PresetRecord`, `FinalistRecord`, `WinnerRecord`, `BundleManifest`, `Claim`,
  `StepState`, `RunManifest`.
- **Contracts:** `Protocol`s above; all I/O returns typed models, never bare dicts.

All persisted records carry a `schema_version` and a complete `inputs` digest.

## 6. Repository layout

```
panther/
├── pyproject.toml              # uv workspace root + shared tool config
├── uv.lock
├── README.md                   # v2 entry point
├── .gitignore                  # ignores panther-old/, data/, build outputs
├── .containerignore
├── docs/
│   ├── rewrite-plan.md         # this document
│   ├── decisions/              # ADR-0001 … ADR-00NN
│   └── current/                # living state, updated every phase
├── packages/
│   ├── panther-core/
│   ├── panther-data/
│   ├── panther-measure/
│   ├── panther-quant/
│   ├── panther-search/
│   ├── panther-pipeline/
│   ├── panther-ops/
│   └── panther-dashboard/
├── apps/
│   └── panther-cli/
├── tests/
│   ├── unit/                   # cross-package unit tests (package-local tests also allowed)
│   ├── integration/            # step wiring against fakes
│   └── parity/                 # old-vs-new golden comparisons + fixtures
├── configs/                    # schemas + default YAML
├── deploy/                     # Containerfile, host/proxy setup, uninstall, nginx
├── pi/                         # Raspberry Pi verifier
└── panther-old/                # git-ignored reference implementation
```

Each package has its own `pyproject.toml`, `src/panther_<name>/`, and `tests/`. The workspace root
declares members and shared tool settings.

## 7. Phases

Each phase lists: goal, tasks, deliverables, exit criteria, and docs. Work is merged to `lalit`
phase by phase.

### M0 — Bootstrap

**Goal:** a clean, green, typed workspace with the documentation system in place.

**Tasks**
1. Rewrite `.gitignore` (add `panther-old/`, data/build outputs) and `.containerignore`.
2. Create the uv workspace, the eight package members and `apps/panther-cli`.
3. Add tooling: ruff, pyright `strict`, pytest, hypothesis, pytest-cov, import-linter, pydantic.
4. Add CI workflow: lint, type-check, unit tests, import contracts.
5. Create `docs/decisions/`, `docs/current/`, seed the bug ledger, add ADRs 0001–0011.
6. Capture reference fixtures from `panther-old` into `tests/parity/fixtures/`.
7. Delete the stale wrapper `.containerignore`/`.gitignore` paths (bug B20).

**Deliverables:** installable workspace, green `uv run pytest`, green `ruff`, green `pyright`,
CI workflow, docs skeleton, first commit on `lalit`.

**Exit criteria:** `uv sync && uv run pytest && uv run ruff check && uv run pyright` all pass;
CI is green on `lalit`; every locked decision has an ADR.

### M1 — `panther-core`

**Goal:** the pure domain engine, fully typed and tested, with the correctness bugs fixed.

**Tasks**
1. Types/bits-per-weight from `gguf`; `Matrix`, `ModelGeometry`; exact byte accounting.
2. `RamModel` (norms/biases + matrices + KV cache + overhead) from typed `SearchConfig`.
3. `Genome`, `Assignment`, `RamWindow`, repair/mutation/crossover primitives.
4. Score math: flips, KL, relative score, paired SE; **fix B4** with block/chunk-aware
   uncertainty and an explicit dependence term.
5. Offline error: **fix B3** — per-input importance for every matrix, correct `(rows, cols)`
   orientation, and an explicit no-imatrix fallback.
6. `CacheKey` builder hashing every input (files, llama commit, constants, config, schema).
7. Preset nominal bpw: **fix B15** by deriving from actual quantize output, not a hand table.

**Deliverables:** `panther-core` with ≥95% line coverage, property-based tests for genome
invariants and statistics, golden tests for formulas that must not change.

**Exit criteria:** all core tests pass; hypothesis finds no window/invariant violations; parser and
statistics regressions for B3/B4/B15 pass.

### M2 — `panther-data`

**Goal:** validated configs and a split-integrity check that actually protects the measurements.

**Tasks**
1. pydantic schemas for `configs/models.yaml` and `configs/search.yaml`.
2. Prompt builder (OpenAssistant, pinned revision) with deterministic splitting.
3. Manifest model + hash verification.
4. **Fix B2:** check the exact files the measurement layer reads (`llama_tasks/selection`,
   `llama_tasks/poc_test`), not a parallel `tasks/` tree.
5. **Fix B19:** require manifest set equality with the files present/used.

**Deliverables:** `panther-data`, synthetic split trees, adversarial overlap tests.

**Exit criteria:** a held-out question placed in selection is detected; a missing/extra manifest
file is detected; configs fail loudly with field-level errors.

### M3 — `panther-measure`

**Goal:** every llama.cpp interaction behind a typed backend with a correct cache.

**Tasks**
1. `LlamaBackend` protocol + subprocess implementation.
2. Port the four output parsers; fuzz them against fixtures.
3. Result cache keyed by a complete `CacheKey`; **fix B1** end-to-end and **B17** (GSM8K key).
4. **Fix B12:** eliminate the `llama-server` port race (hold the listener / hand off ownership /
   verify the server's identity).
5. **Fix B13:** one thread policy derived from runtime config.
6. Reference generation (chat answers, `.kld` bases, correctness records).

**Deliverables:** `panther-measure`, fake backend, parser fuzz tests, cache invalidation matrix.

**Exit criteria:** changing any input invalidates exactly the dependent results; the port-race
regression test passes under stress; parsers reject malformed logs.

### M4 — Vertical slice (search + pipeline + CLI skeleton)

**Goal:** one contest end-to-end on fixtures, proving the step graph and cache.

**Tasks**
1. `panther-search`: preset selection, GA, CP-SAT proposals.
2. **Fix B5** (first-look margin from SE), **B6** (no ratchet), **B7** (regularized solver),
   **B8** (deterministic scheduling).
3. `panther-pipeline`: ordered step graph, resumability, invalidation, final/guard/verdict,
   versioned bundle manifest.
4. `panther-cli` skeleton: `run`, `stop`, `status` wired to fakes; `--dry-run`.
5. End-to-end test: `1.5b@1` runs fetch→publish on a temp data dir with fake store/llama/HF.

**Deliverables:** runnable vertical slice, reproducibility test, invalidation matrix.

**Exit criteria:** same seed and fixtures produce identical records; re-running is a no-op;
changing one input invalidates exactly the right steps; the winner record validates.

### M5 — `panther-quant`

**Goal:** the real, numerically correct quantized store and distillation.

**Tasks**
1. HF fetch/backup behind `HuggingFaceRepo`; resume support.
2. Store builder: 16-bit GGUF, importance matrix, 14 type copies.
3. GPTQ rounding (port the ctypes `ggml_quantize_chunk` interface; golden-test numerics).
4. Distillation (scales + norms; row reorder; IQ1_M left untouched).
5. **Fix B10** (disk preflight + stale-file cleanup), **B18** (typed byte/GPU planner),
   **B11** (measure and guard the 24B load; GPU-marked).
6. **Fix B21** (remove dead constants).

**Deliverables:** `panther-quant`, tiny synthetic GGUF fixtures, numerical golden tests.

**Exit criteria:** GPTQ output matches the reference implementation within tolerance on a fixed
matrix; distillation writes valid GGUF with unchanged size; disk preflight refuses impossible runs.

### M6 — `panther-ops`

**Goal:** typed CLI and trustworthy resource/container orchestration.

**Tasks**
1. `ContainerRuntime` protocol + podman implementation; typed container specs.
2. Claim manager: **fix B9** (caps 96–100 GiB configurable), **B16** (single inflight planner).
3. Locks and the run-lock handoff, ported and tested.
4. CLI: `run`, `stop`, `pause`, `resume`, `status`, `build`, `test`, `dashboard`, `exec`, `cexec`.
5. GitLab guard and `PAUSED`/`PAUSED.guard` semantics.
6. **Fix B14:** a single source of truth for caps, rendered into docs/dashboard.

**Deliverables:** `panther-ops` + `panther-cli`, recording fake runtime, arg snapshots.

**Exit criteria:** claim fit/refusal/lapse and lock handoff tests pass; no hard-coded cap appears
outside the config.

### M7 — Dashboard

**Goal:** the same read-only observability, typed and contract-tested.

**Tasks**
1. Typed state assembly from `data/` + masked site names.
2. `/api/state`, `/api/logs`, `/api/logs/stream` preserved; SSE limits.
3. Decide and record: reuse the old static frontend (parity) vs rebuild (modern). **ADR required.**
4. **Fix B14** in rendered numbers.

**Deliverables:** `panther-dashboard`, fixture data dir, contract tests.

**Exit criteria:** `/api/state` validates against its schema on fixtures; masking is proven;
streams cap at 16 and release on disconnect.

### M8 — Raspberry Pi verifier

**Goal:** the same winner check with a typed manifest and queue.

**Tasks**
1. `BundleManifest` model shared with `panther-pipeline`.
2. Queue with poll/download/verify/run/report/delete.
3. FITS/OVER, speed and throttling reporting; failure lines preserved.

**Deliverables:** `pi/` rewritten with tests against a fake `llama-server`.

**Exit criteria:** checksum mismatch, early exit, unhealthy server and upload failure are all
covered by tests and produce the documented result lines.

### M9 — Deploy

**Goal:** reproducible image and reversible host setup.

**Tasks**
1. Containerfile at the pinned llama.cpp commit and CUDA base (reuse the old build).
2. `host-setup.sh`, `proxy-setup.sh`, `uninstall.sh`, nginx configs; ported with a ledger.
3. Shellcheck/bats tests; `--dry-run` goldens.

**Deliverables:** `deploy/` with tests.

**Exit criteria:** dry runs are idempotent; the uninstall ledger replays cleanly; nginx config
passes `nginx -t`.

### M10 — Parity and end-to-end

**Goal:** prove old-vs-new equivalence and close the bug ledger.

**Tasks**
1. Golden parity harness: run both pipelines on captured fixtures and diff typed records.
2. Document every intentional difference with its ADR/bug entry.
3. Opt-in GPU/network smoke run (nightly or manual).
4. Coverage gates and mutation testing on core where practical.

**Deliverables:** parity report, closed bug ledger, coverage report.

**Exit criteria:** all ledger items have a passing regression test; parity diffs are either zero
or explained; a GPU smoke run completes one contest.

### M11 — Cutover

**Goal:** declare v2 the implementation of record.

**Tasks**
1. Final `docs/current/` update; archive the old plan.
2. Tag `v1.0.0` (or the agreed version).
3. Keep `panther-old/` as a frozen, git-ignored reference.

**Exit criteria:** green CI, complete docs, tagged release.

## 8. Bug-fix ledger

Every defect from the code audit is tracked here. A bug is **closed** only when its fix is merged
and its named regression test passes.

| ID | Sev | Old location | Problem | Fix | Phase | Regression test |
|---|---|---|---|---|---|---|
| B1 | High | `search.ga_ins`, `preset_ins`, `final_ins`, `measure.fits`, labels | Cache keys omit inputs (KL weight, question counts, window, RAM model, llama commit, file hashes, verdict rules) | Content-addressed `CacheKey` over every input + versioned records | M1/M3/M4/M6 | `test_cache_key_invalidation_matrix` |
| B2 | High | `data.check` | Checks `tasks/` but measures `llama_tasks/`; never checks measured selection vs held-out overlap | Check the exact measured files; verify overlap and manifest completeness | M2 | `test_integrity_checks_measured_files` |
| B3 | High | `search._error`, `_error_one` | Offline error uses ones for no-imatrix matrices and sums/weights the wrong axes | Per-input importance everywhere; `(rows, cols)` orientation; explicit fallback | M1 | `test_offline_error_matches_formula` |
| B4 | High | `measure.score_se`, `relative` | Tokens treated as independent; our/preset KL treated as independent though they share text | Block/chunk-aware SE + explicit covariance/dependence term | M1/M3 | `test_score_interval_under_dependence` |
| B5 | Med | `search.DROP_MARGIN` | Hand-set 3.0 ≈ 1.2 SE, not ~2 SE | Derive from measured SE at the current flip rate | M4 | `test_first_look_cut_from_se` |
| B6 | Med | `search.ga` stop rule | Ratchets: sub-SE gains still lower `best`, so patience never resets | Compare against the best at the last counted improvement | M4 | `test_patience_does_not_ratchet` |
| B7 | Med | `search.Genes.proposals` | ~30 weights from 8–70 genomes; clipped weights collapse groups | Regularized / shrunk fit; constrain solver | M4 | `test_solver_stability` |
| B8 | Med | `search.ga` | `as_completed` order makes runs non-reproducible | Deterministic scheduling + seeded RNG | M4 | `test_ga_is_deterministic` |
| B9 | High | `run.CAP_RAM` | 120 GiB of a 124 GB host, ~4 GiB left for GitLab | 96–100 GiB, configurable, documented | M6 | `test_claim_caps` |
| B10 | Med | `store.MARGIN` | 20 GiB margin too small for 22B/24B stores + references | Preflight planner + stale-file cleanup | M5 | `test_disk_preflight` |
| B11 | Med | `quant`/`distill` | 24B transformers load may exceed the claim (~94 vs ~52 GiB) | Measure real peak; enforce a budget guard | M5/M10 | `test_24b_load` (gpu) |
| B12 | Med | `measure.complete` | Port is released before `llama-server` binds it | Hold the listener / hand off / verify identity | M3 | `test_server_port_race` |
| B13 | Low | `run` + `measure` | `PANTHER_CPUS` used as both container cap and llama `-t` on 48 threads | One thread policy from runtime config | M3/M6 | `test_thread_policy` |
| B14 | Low | docs + dashboard | Hard-coded 92 vs actual 94 GiB; stale reserve numbers | Single source of truth rendered into docs/UI | M6/M7 | `test_caps_have_one_source` |
| B15 | Low | `search.PRESETS` | `Q2_K` nominal bpw 3.2 vs real ~2.63; can skip fitting presets | Derive bpw from actual quantize output / corrected table | M1/M4 | `test_preset_bpw_table` |
| B16 | Low | `run.sizes` vs `ga` | Inflight count computed twice from different inputs | One typed planner | M6 | `test_inflight_planner` |
| B17 | Low | `measure.gsm8k` | Result key has no question/few-shot/llama-commit hash | Include all of them in the key | M3 | `test_gsm8k_cache_key` |
| B18 | Low | `store.build` | Disk bytes and GPU bytes mixed in one heuristic | Typed byte/GPU planner | M5 | `test_rounding_plan` |
| B19 | Low | `data.check` | A manifest that omits a file passes | Require exact set equality | M2 | `test_manifest_completeness` |
| B20 | Low | wrapper `.containerignore`/`.gitignore` | Stale paths, nested wrapper directory | Rewrite at repo root; remove wrapper | M0 | `test_repo_hygiene` |
| B21 | Low | `run.RUN_RAM` | Dead constant | Remove; lint dead code | M0 | lint |

## 9. Testing strategy

| Tier | Runs in CI | What it covers | Tooling |
|---|---|---|---|
| **Unit** | yes, every push | pure domain, parsers, statistics, genome invariants, cache keys, schemas | pytest, hypothesis |
| **Integration** | yes, every push | step wiring with fake backends in a temp data dir | pytest, tmp_path, fakes |
| **Parity** | yes, every push | old-vs-new golden records; intentional diffs asserted | pytest + captured fixtures |
| **GPU** | opt-in (`-m gpu`) | real llama.cpp, GPTQ numerics, distillation, 24B load | pytest marker, GPU runner |
| **Network** | opt-in (`-m network`) | HF fetch/publish, oasst1 prompts | pytest marker |
| **Slow** | opt-in (`-m slow`) | full contest smoke | pytest marker |

Rules:
- No test may touch the real network, GPU, `~/.cache`, or the repo's `data/` unless marked.
- Every bug in §8 has a regression test named in the ledger.
- Coverage gates: core/data ≥95%, measure/search/pipeline ≥90%, ops/dashboard/pi ≥80%.
- Fixtures are small, deterministic and committed; large artifacts are generated in `tmp_path`.

## 10. Tooling and dependency policy

**Always reuse an existing module.** New code is reserved for Panther-specific logic.

| Need | Use |
|---|---|
| Packaging / env | `uv`, `hatchling` |
| CLI | `typer` (+ `rich` for output) |
| Config / models | `pydantic` v2, `pydantic-settings`, `pyyaml` |
| Logging | stdlib `logging` + a small JSON formatter |
| HTTP client | stdlib `urllib` or `httpx` |
| HTTP server | stdlib `http.server` (parity with old dashboard) |
| Hashing / atomic files / locks | stdlib `hashlib`, `os.replace` + `tempfile`, `fcntl` |
| Subprocess | stdlib `subprocess` |
| Math | `numpy` |
| GGUF | `gguf` |
| Optimization | `ortools` (CP-SAT) |
| Quantization / distillation | `torch`, `transformers`, `accelerate` |
| HF | `huggingface_hub` |
| Parquet prompts | `pyarrow` |
| Lint / format | `ruff` |
| Types | `pyright` (`strict`) |
| Import boundaries | `import-linter` |
| Tests | `pytest`, `hypothesis`, `pytest-cov` |

No dependency is added without an ADR if it is not in this table.

## 11. Git and CI workflow

- **Remote:** `origin` = `https://github.com/RachitKumar205/panther.git`.
- **Branch:** all work on `lalit`; never commit to `master`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`).
  One phase may be several commits; each must leave the tree green.
- **Push:** `gh` for repository operations; `git push origin lalit` for code.
- **CI:** on every push to `lalit`: ruff, pyright, import-linter, unit + integration + parity tests.
- **PRs:** optional; when used, target `master` and summarize the phase + ledger items closed.
- **Tags:** `v1.0.0` at cutover.

## 12. Documentation system

### ADRs — `docs/decisions/NNNN-slug.md`

Template:

```markdown
# NNNN — Title

- **Status:** proposed | accepted | superseded by NNNN
- **Date:** YYYY-MM-DD
- **Phase:** M0…

## Context
What forced the decision.

## Decision
What we will do.

## Alternatives considered
What else was on the table and why it lost.

## Consequences
What becomes easier/harder; what we must now maintain.

## Links
Code, tests, and the old finding (B#) this resolves.
```

### Living docs — `docs/current/`

| File | Contents |
|---|---|
| `architecture.md` | package graph, protocols, data flow |
| `packages.md` | each package's responsibility and public API |
| `data-model.md` | configs, records, artifacts, manifests, schema versions |
| `pipeline.md` | steps and which split each reads |
| `testing.md` | tiers, markers, fixtures, coverage gates |
| `operations.md` | containers, claims, locks, guard, deploy |
| `observability.md` | dashboard and Pi |
| `status.md` | phase progress and open work |
| `bug-fix-ledger.md` | live copy of §8 with status |
| `glossary.md` | terms |

`docs/current/` is updated in the same commit as the change it describes.

## 13. Risks

| Risk | Mitigation |
|---|---|
| torch/transformers wheels for 3.12 + CUDA 12.9 | pin via uv; `panther-quant` is the only torch importer |
| ctypes/GPTQ numerical drift | tiny synthetic GGUF golden tests; compare with old `rounding.py` |
| llama.cpp output format drift | parsers fuzzed; pinned commit recorded in every cache key |
| No old outputs to diff | capture fixtures from `panther-old` in M0 before changing anything |
| Rewrite drifts from the science | parity harness + ledger + ADRs |
| Ops/deploy scope creep | ported late behind `ContainerRuntime`; bash stays where it is the right tool |
| 24B memory blow-up | measured in M5/M10; guard refuses an impossible claim |

## 14. Definition of done

A phase is done when:

1. Its code is typed (`pyright strict`), linted (`ruff`) and documented (docstrings + `docs/current`).
2. Its unit, integration and (where relevant) parity tests pass in CI.
3. Every bug it closes has a regression test and the ledger is updated.
4. Any decision it required has an ADR.
5. The working tree is green and pushed to `lalit`.

## 15. Execution checklist

- [x] Decisions agreed (ADRs 0001–0011)
- [x] `lalit` branch created and based on `master`
- [x] `panther-old/` added to `.gitignore`
- [x] Rewrite plan written
- [ ] M0 bootstrap (workspace, tooling, CI, docs, fixtures)
- [ ] M1 `panther-core`
- [ ] M2 `panther-data`
- [ ] M3 `panther-measure`
- [ ] M4 vertical slice
- [ ] M5 `panther-quant`
- [ ] M6 `panther-ops` + CLI
- [ ] M7 dashboard
- [ ] M8 Pi
- [ ] M9 deploy
- [ ] M10 parity + end-to-end
- [ ] M11 cutover
