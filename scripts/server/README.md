# scripts/server: the dashboard

A read-only web page of the pipeline. `./panther dashboard` (re)starts it on the server in its own container, with
`data/` mounted read-only; `./panther run` starts it if it is not running.

**What it reads:** the logs (`data/logs/panther.log`, `data/logs/progress-b<B>.log`), each contest's winner record
(`data/rewrite/budget-B/winner-<model>-v2.json`), the active runs' claims (`data/locks/claims.json` and the
`run-<id>.lock` files), `configs/search.yaml` (the planned contests), GPU memory (`nvidia-smi`) and disk, and the Pi's
results from Hugging Face (below). `deploy/server/local.env` gives the names it masks.

`dashboard.py` serves the page and `/api/state` (everything it shows, as JSON); no fonts or scripts come from the
internet. A sidebar of sections, list pages that lead to detail pages, tabs on a detail page, and links that can be
shared (`#/contests/22b@5/steps`).

| Page | Shows |
|---|---|
| Overview | contests won, what is running and when it ends, GPU and disk; every verdict; recent activity |
| Contests | every contest, filterable (won, tied, running, failed or stopped, dropped) and sortable; one row each |
| Contest | Results (the relative score with its 95 % interval, 100 = the preset, lower is better; older results: ours − preset in accuracy points; held-out table by task, accuracy kept, the verdict at KL weight 5 and 20, the Pi's RAM with FITS / OVER and speed, our mix of types), Steps (a timeline of the pipeline's steps with start times and durations; the step a failed or stopped run was in shows as such), Events |
| System | GPU memory, the active runs' claims (`data/locks/claims.json`; a claim shows while its `run-<id>.lock` is held, "unknown" if the lock cannot be read), disk, the Pi's measurements (a failed Pi check shows as failed), files that could not be read |
| Logs | a live tail of every log (the last 3,000 lines, then each new line about a second after it is written, pushed by the server as server-sent events): search, level (key, notes, errors), which log, wrap, follow; scroll up and it holds still and counts new lines |

A file that cannot be read (bad JSON, a Pi line without its numbers) is named in a banner and on the System page;
the rest of the page still shows. It refreshes every 10 s; jobs are controlled only with `./panther`. Since it may be
public, the server's account and address and the shared service's domain are masked in everything it sends, and at
most 16 live log streams run at once. Where it listens on the server: [deploy/server](../../deploy/server/README.md).

```bash
uv run python scripts/server/dashboard.py --data /path/to/data --port 43800   # then open http://localhost:43800
```

`dashboard/` holds the page (HTML, CSS, JS with hand-drawn SVG charts) and its font, Recursive (SIL Open Font
License, `dashboard/OFL.txt`). The Pi's results: the dashboard fetches `pi/<contest>.txt` from the public Hugging Face
repo `<user>/panther-status` (`<user>`: the part of `PANTHER_HF_REPO` before `/`) every 5 minutes into its `/tmp`,
over the older `data/pi/*.txt`; a failed fetch is shown on the System page and the last good results are kept.
