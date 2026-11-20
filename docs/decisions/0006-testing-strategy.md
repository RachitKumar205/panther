# 0006 — Testing strategy: fakes in CI, real tiers opt-in

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M0

## Context

The pipeline depends on a GPU, `llama.cpp`, `torch`, Hugging Face and rootless podman. Requiring
all of that in CI would make the suite slow, flaky and machine-specific. Testing only with mocks
would miss integration defects.

## Decision

- **CI (every push):** unit, integration against fakes, and golden parity tests. No GPU, network or
  containers.
- **Opt-in tiers:** `@pytest.mark.gpu`, `@pytest.mark.network`, `@pytest.mark.slow`, run manually
  and nightly.
- Every external dependency is behind a typed `Protocol` with a real and a fake implementation.
- Coverage gates: core/data ≥95%, measure/search/pipeline ≥90%, ops/dashboard/pi ≥80%.

## Alternatives considered

- **Real GPU in CI.** Rejected: cost and flakiness.
- **Fakes only.** Rejected: no proof the real backend works.

## Consequences

The whole pipeline can be exercised deterministically on a laptop; real-hardware confidence comes
from the opt-in tier and the M10 smoke run.

## Links

- `docs/rewrite-plan.md` §9
- ADR 0003, ADR 0009
