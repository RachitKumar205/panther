# Panther v2

Panther builds a llama.cpp GGUF file for each (model, RAM-budget) **contest** that fits a
Raspberry Pi 5 and loses less quality than llama.cpp's best ready-made preset that also fits.

This repository is a from-scratch, typed rewrite of the existing pipeline. It is under active
development on the `lalit` branch.

- **Rewrite plan (source of truth):** [docs/rewrite-plan.md](docs/rewrite-plan.md)
- **Decisions (ADRs):** [docs/decisions/](docs/decisions/)
- **Current state:** [docs/current/](docs/current/)

The previous implementation is preserved as a read-only reference in `panther-old/`
(git-ignored) and on the old `master` branch.
