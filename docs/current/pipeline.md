# Pipeline

> Target pipeline. This mirrors the old steps so parity is measurable; each step is reimplemented
> in its phase and gains a complete input digest.

| Step | Implemented in | Reads | Writes |
|---|---|---|---|
| fetch | `panther-quant` store | HF backup or original weights | `store/<model>/` |
| prompts | `panther-data` | OpenAssistant oasst1 | `splits/text/prompts.json` |
| claim | `panther-ops` | `claims.json`, live locks | `claims.json` |
| check | `panther-data` | split manifests, measured task files | — (stops the run on failure) |
| store | `panther-quant` | BF16 original, imatrix, train web | 14 type copies |
| error | `panther-core`/`panther-data` | type copies, imatrix | `error.json` |
| refs | `panther-measure` | chat prompts, select/test text | chat answers, `.kld` bases |
| preset | `panther-search` | HF original, imatrix | preset files + record |
| ga | `panther-search` | preset record, error, select data | `list-<model>.json` |
| final | `panther-pipeline` | finalists, preset, train/select/test | `winner-<model>.json` |
| bundle | `panther-pipeline` | winner file | `pi_bundle/...` |
| upload | `panther-cli` | bundle | HF repo `<user>/panther-<size>-<budget>g-v2` |
| tidy | `panther-ops`/`panther-quant` | claims, store | backups, cleanup |

## Splits

| Stage | train | select | test |
|---|---|---|---|
| store: imatrix, rounding | web | | |
| refs: original probabilities | | chat+web | chat+web |
| ga: KL | | chat+web | |
| final: distillation | chat+web | | |
| final: winner/guard | | chat+web | |
| final: held-out KL/PPL | | | chat+web |
| benchmark: preset/ga/final | | selection questions | |
| benchmark: verdict | | | held-out questions |

The `check` step must verify the **measured** task files (`llama_tasks/selection`,
`llama_tasks/poc_test`), not a parallel `tasks/` tree (bug B2).
