# scripts/server

`pipeline.py` runs one contest (a model at a RAM budget) inside the server's containers; the project README's "How
it works" explains every step. Each step skips work already on disk and writes progress lines to
`data/logs/progress-b<BUDGET>.log`.

| Step | Output |
|---|---|
| `fetch` (CPU container) | `data/hf/<model>` |
| `store` | `data/store/<model>/`: 16-bit original, importance matrix, one type copy per type (14), error-compensated |
| `error` | `rewrite/<model>/error.json` |
| `refs` | `data/reference/<model>/{selection,answers}.kld`, the original's answers and question results |
| `preset` | `rewrite/budget-B/preset-<model>.json` |
| `ga` | `rewrite/budget-B/list-<model>-v2.json` (top 3), `ga-<model>-v2.json` (every genome), `ga-<model>-live.json` |
| `final` | `rewrite/<model>/tuned/` (the distilled top 3), `rewrite/budget-B/winner-<model>-v2.json` (winner, verdict, held-out numbers for ours, preset, original) |
| `bundle`, `upload` | `data/pi_bundle/budget-B-<size>-v2/` → private Hugging Face repo `panther-<size>-<B>g-v2` |
| `qatdata`, `qat` (`./panther qat C`, an experiment) | `rewrite/<model>/qat/`: token samples, the rebuilt undistilled winner, Stage 1, Stage 1 + 2, Stage 2 only; `rewrite/budget-B/qat-<model>.json` (held-out results vs the winner and the preset) |

`./panther start` (repo root) runs every contest through these steps; `pipeline.py run B MODEL` runs store → bundle
in order inside one GPU container. `status.py` rewrites `data/status.md` every 30 s
(verdicts, every contest's steps with time spent and left, the Pi's results) and publishes it to
`<user>/panther-status` every 60 s.


Every formula and number (RAM model, rounding, score, search, distillation, verdict): the project README,
[section 6](../../README.md#6-in-depth-every-number-and-formula).

## The dashboard

`dashboard.py` serves a self-hosted, read-only web page of the whole pipeline, from `data/` only (no outside
service, no fonts or scripts from the internet): every contest (state, steps and ETA, the verdict as a forest plot
of ours − preset with its 95% interval, the full held-out table, our mix of types, the Pi's RAM and speed), the
EfficientQAT experiment (Stage 1's strategy, every block's validation error before and after, each block's training
curve, Stage 2's KL with a trend line, the held-out results against the winner and the preset), GPU memory and
bookings, disk, and the log. It refreshes every 10 s; jobs are still controlled with `./panther`.

```bash
uv run python scripts/server/dashboard.py --data /path/to/data --port 43800   # then open http://localhost:43800
```

`dashboard/` holds the page (HTML, CSS, JS with hand-drawn SVG charts) and its font, Recursive (SIL Open Font
License, `dashboard/OFL.txt`). The Pi's results reach `data/pi/` through `status.py`.
