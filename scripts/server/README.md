# scripts/server

`pipeline.py` runs one contest (a model at a RAM budget) inside the server's containers; the project README's "How
it works" explains every step. Each step skips work already on disk and writes progress lines to
`data/logs/progress-b<BUDGET>.log`.

| Step | Output |
|---|---|
| `fetch` (CPU container) | `data/hf/<model>` |
| `store` | `data/store/<model>/`: 16-bit original, importance matrix, one type copy per menu type |
| `error` | `rewrite/<model>/error.json` |
| `refs` | `data/reference/<model>/{selection,answers}.kld`, the original's answers and question results |
| `preset` | `rewrite/budget-B/preset-<model>.json` |
| `ga` | `rewrite/budget-B/list-<model>-v2.json` (top 3), `ga-<model>-v2.json` (every genome), `ga-<model>-live.json` |
| `final` | `rewrite/budget-B/winner-<model>-v2.json` (winner, verdict, held-out numbers for ours, preset, original) |
| `bundle`, `upload` | `data/pi_bundle/budget-B-<size>-v2/` → private Hugging Face repo `panther-<size>-<B>g-v2` |

`python pipeline.py run B MODEL` runs store → bundle in order. `status.py` rewrites `data/status.md` every 30 s
(verdicts, every contest's steps with time spent and left, the Pi's results) and publishes it to
`<user>/panther-status` every 60 s.
