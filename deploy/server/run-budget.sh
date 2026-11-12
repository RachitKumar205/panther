#!/usr/bin/env bash
# The whole standard (docs/STANDARD.md) for one RAM budget, unattended, in one command. Run on the server:
#
#   deploy/server/run-budget.sh 7
#
# One lane per model, all in parallel. A searched model goes straight through:
#   download -> type copies -> offline error -> the original's reference outputs -> llama.cpp presets (the best one
#   is the competitor and joins the GA's start) -> genetic search (fitness: accuracy on selection questions, KL as a
#   tie-break) -> top genomes on all selection questions + guard against the real preset -> full test of the winner
#   -> winner vs preset -> Pi bundle -> upload
# (to a private Hugging Face repo, PANTHER_HF_REPO in local.env; the Pi downloads it from there)
# without waiting for the other models. A reference model only needs its uniform Q6_K file, then its tests.
# When every lane is done: pick the winners, bundle them for the Pi.
# Idempotent: every step checks the disk and skips work that is already there; a step already running is adopted.
# Several budgets can run at the same time: steps are named per budget, and steps that write a model's shared files
# (quantized copies, offline error, references, part costs) take a lock per model, so the second waits and skips.
# Heavy steps wait while the data disk has less than 30 GB free.
set -euo pipefail
B="${1:?usage: run-budget.sh BUDGET_GIB}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="${PANTHER_WORK:-/s3/panther/work}"
P="$W/data/rewrite/chains"; mkdir -p "$P"
R="/opt/venv/bin/python scripts/server/pipeline.py"
line="$(grep -E "^  $B: " "$ROOT/configs/search.yaml")" || { echo "budget $B not in configs/search.yaml" >&2; exit 1; }
list() { echo "$line" | sed -E "s/.*$1: \[([^]]*)\].*/\1/" | tr -d ' ' | tr ',' ' '; }
short() { echo "$1" | sed -E 's/^[^-]+-([0-9.]+B).*/\1/' | tr 'A-Z' 'a-z'; }
running() { local ps_out; ps_out="$(ps -eo args)"; grep -qE "chain.sh $1 /|after-chains.sh .* $1 /" <<< "$ps_out"; }  # the lane's own runner only
start() { running "$2" && { echo "lane $2 already running"; return; }
          PANTHER_CPUS="${PANTHER_CPUS:-16}" setsid nohup "${@:3}" >/dev/null 2>&1 < /dev/null & echo "started $2"; }
lanes=""
for m in $(list searched); do
  s="$(short "$m")"; c="g$B-$s"; lanes="$lanes $c"
  running "$c" && { echo "lane $c already running"; continue; }  # never rewrite a running lane's steps
  # steps are named by method (rg-: genetic search), so a finished step of another method is never taken as done
  printf "%s\n" "cpu rg-fetch-$s bash scripts/server/fetch_model.sh $m" "gpu rg-store$B-$s $R store $B $m" \
    "gpu rg-err$B-$s $R error $B $m" "gpu rg-refs-$s $R refs $m" "gpu rg-preset$B-$s $R preset $B $m" \
    "gpu rg-ga$B-$s $R ga $B $m" "gpu rg-gascore$B-$s $R gascore $B $m" "gpu rg-vspreset$B-$s $R preset $B $m" \
    "gpu rg-bundle$B-$s $R bundle $B $m" "cpu rg-upload$B-$s $R upload $B $m" > "$P/$c.steps"
  start x "$c" "$HERE/chain.sh" "$c" "$P/$c.steps"
done
echo "budget $B: lanes${lanes}. Logs: $W/data/logs/chain-*.log, progress in $W/data/logs/progress-b$B.log"
