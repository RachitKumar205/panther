#!/usr/bin/env bash
# Start contests one after another as room frees up. Run on the server, detached:
#
#   setsid nohup deploy/server/queue.sh "7 InternLM2.5-20B-Chat" "5 Qwen2.5-14B-Instruct" >/dev/null 2>&1 &
#
# The next "BUDGET MODEL" starts (run-budget.sh) when fewer than MAX_LANES lanes run and at least MIN_FREE_GB is
# free. When no lane of a model is running or queued any more, its type copies are deleted: only the original, its
# importance matrix and the presets stay on disk. Log: data/logs/queue.log
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${PANTHER_WORK:-/s3/panther/work}"
D="$W/data"
MAX_LANES="${MAX_LANES:-3}"; MIN_FREE_GB="${MIN_FREE_GB:-80}"
LOG="$D/logs/queue.log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
lanes() { ps -eo args | grep -E "^bash [^ ]*chain.sh g[0-9]+-" | awk '{print $3}'; }
lane_model() { awk '/ ga /{print $NF}' "$D/rewrite/chains/$1.steps" 2>/dev/null; }
clean() {  # type copies of models with no running lane and nothing queued
  for f in "$D"/store/*/*-*.gguf; do
    m="$(basename "$(dirname "$f")")"; t="$(basename "$f" .gguf)"; t="${t#"$m"-}"
    [ "$t" = BF16 ] && continue
    [[ "$(basename "$f")" == preset-* ]] && continue
    for l in $(lanes); do [ "$(lane_model "$l")" = "$m" ] && continue 2; done
    grep -qw -- "$m" <<< "$*" && continue
    for c in "$D"/logs/chain-g*.log; do  # a lane that stopped on a failure may be restarted: keep its copies
      l="$(basename "$c" .log)"; l="${l#chain-}"
      [ "$(lane_model "$l")" = "$m" ] && tail -n 1 "$c" | grep -q "stopped at" && continue 2
    done
    rm -f "$f" && log "deleted type copy $(basename "$f") ($m has no running or queued lane)"
  done
}
queue=("$@")
log "queue: ${queue[*]}"
while :; do
  clean "${queue[@]:-}"
  [ ${#queue[@]} -eq 0 ] && { [ -z "$(lanes)" ] && break; sleep 120; continue; }
  free="$(df --output=avail -B1G "$D" | tail -1 | tr -d ' ')"
  if [ "$(lanes | wc -l)" -lt "$MAX_LANES" ] && [ "$free" -ge "$MIN_FREE_GB" ]; then
    read -r B M <<< "${queue[0]}"
    log "starting $B GiB $M (${free} GB free)"
    "$HERE/run-budget.sh" "$B" "$M" >> "$LOG" 2>&1
    queue=("${queue[@]:1}")
  fi
  sleep 120
done
log "queue done"
