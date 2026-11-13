#!/usr/bin/env bash
# The one entrypoint: run contests on the server, up to J at once, unattended.
#
#   setsid nohup deploy/server/run.sh -j 2 "3 Qwen2.5-7B-Instruct" "5 Qwen2.5-14B-Instruct" >/dev/null 2>&1 &
#
# Each contest: fetch (CPU job, network), every pipeline step (GPU job; GPU memory is shared through a booking
# ledger), upload for the Pi (CPU job). Steps skip finished work, so a stopped contest is simply started again.
# At the end the type copies of these models are deleted unless another run is using the model.
# Log: data/logs/run.log; progress: the status page.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${PANTHER_WORK:-/s3/panther/work}"
LOG="$W/data/logs/run.log"
R=(/opt/venv/bin/python scripts/server/pipeline.py)
J=1; [ "${1:-}" = -j ] && { J="$2"; shift 2; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
job() {  # job cpu|gpu NAME CMD...: run, wait, 0 if it exited 0
  local kind="$1" name="$2"; shift 2
  "$HERE/panther.sh" "${kind/cpu/}job" "$name" "$@" >> "$LOG" 2>&1 || return 1
  [ "$("$HERE/panther.sh" wait "$name" | head -1)" = 0 ]
}
contest() {
  local B M n; read -r B M <<< "$1"
  n="$(echo "$M" | cut -d- -f2 | tr 'A-Z' 'a-z')-b$B-$(date +%s)"
  log "start $B GiB $M"
  if job cpu "fetch-$n" "${R[@]}" fetch "$B" "$M" && job gpu "run-$n" "${R[@]}" run "$B" "$M" \
     && job cpu "upload-$n" "${R[@]}" upload "$B" "$M"; then log "done $B GiB $M"; else log "FAILED $B GiB $M"; fi
}
for c in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
  contest "$c" &
  sleep 2
done
wait
for c in "$@"; do
  M="${c#* }"
  pgrep -f "pipeline.py (run|fetch) [0-9]+ $M" > /dev/null && continue
  find "$W/data/store/$M" -name "$M-*.gguf" ! -name "*-BF16.gguf" -delete && log "deleted the type copies of $M"
done
