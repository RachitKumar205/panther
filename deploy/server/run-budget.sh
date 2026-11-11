#!/usr/bin/env bash
# The whole standard (docs/STANDARD.md) for one RAM budget, unattended, in one command. Run on the server:
#
#   deploy/server/run-budget.sh 7
#
# One lane per model, all in parallel. A searched model goes straight through:
#   format files -> offline error -> references -> part costs -> solver list -> tests -> winner -> Pi bundle
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
short() { echo "$1" | sed -E 's/Qwen2\.5-([0-9.]+B).*/\1/' | tr 'A-Z' 'a-z'; }
running() { local ps_out; ps_out="$(ps -eo args)"; grep -qE "chain.sh $1 |after-chains.sh .* $1 " <<< "$ps_out"; }
start() { running "$2" && { echo "lane $2 already running"; return; }
          PANTHER_CPUS="${PANTHER_CPUS:-16}" setsid nohup "${@:3}" >/dev/null 2>&1 < /dev/null & echo "started $2"; }
lanes=""
for m in $(list searched); do
  s="$(short "$m")"; c="b$B-$s"; lanes="$lanes $c"
  running "$c" && { echo "lane $c already running"; continue; }  # never rewrite a running lane's steps
  printf "%s\n" "gpu rw-store$B-$s $R store $B $m" "gpu rw-err$B-$s $R error $B $m" "gpu rw-refs-$s $R refs $m" \
    "gpu rw-sens$B-$s $R sens $B $m" "gpu rw-list$B-$s $R list $B $m" "gpu rw-test$B-$s $R test $B $m" \
    "gpu rw-pick$B-$s $R pick $B $m" "gpu rw-bundle$B-$s $R bundle $B $m" > "$P/$c.steps"
  start x "$c" "$HERE/chain.sh" "$c" "$P/$c.steps"
done
for m in $(list reference); do
  s="$(short "$m")"; c="b$B-ref-$s"; lanes="$lanes $c"
  running "$c" && { echo "lane $c already running"; continue; }
  printf "%s\n" "gpu rw-store$B-$s $R store $B $m" "gpu rw-test$B-ref-$s $R test $B $m" > "$P/$c.steps"
  start x "$c" "$HERE/chain.sh" "$c" "$P/$c.steps"
done
running "b$B" || printf "%s\n" "gpu rw-pick$B $R pick $B" > "$P/b$B.steps"  # comparison across models
start x "b$B" "$HERE/after-chains.sh" "${lanes# }" "b$B" "$P/b$B.steps"
echo "budget $B: lanes${lanes}; then pick + bundle (chain b$B). Logs: $W/data/logs/chain-*.log"
