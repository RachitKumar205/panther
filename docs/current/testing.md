# Testing

> Tiers and gates. See `docs/rewrite-plan.md` §9 for the full strategy.

## Tiers

| Tier | CI | Marker | Contents |
|---|---|---|---|
| Unit | yes | — | pure domain, parsers, statistics, genome invariants, cache keys, schemas |
| Integration | yes | — | step wiring with fake backends in a temp data dir |
| Parity | yes | — | old-vs-new golden records; intentional diffs asserted |
| GPU | opt-in | `gpu` | real llama.cpp, GPTQ numerics, distillation, 24B load |
| Network | opt-in | `network` | HF fetch/publish, oasst1 prompts |
| Slow | opt-in | `slow` | full contest smoke |

## Rules

- No unmarked test touches the network, GPU, `~/.cache`, or the repo's `data/`.
- Every ledger bug has a named regression test (see [bug-fix-ledger.md](bug-fix-ledger.md)).
- Fixtures are small and deterministic; large artifacts go in `tmp_path`.

## Coverage gates

| Area | Minimum |
|---|---|
| core, data | 95% |
| measure, search, pipeline | 90% |
| ops, dashboard, pi | 80% |
