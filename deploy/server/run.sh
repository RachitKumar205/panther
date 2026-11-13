#!/usr/bin/env bash
# Run contests one after another on the server, unattended (start several for parallel lanes):
#
#   setsid nohup deploy/server/run.sh "5 Qwen2.5-14B-Instruct" "7 Mistral-24B-Instruct" >/dev/null 2>&1 &
#
# Each contest: fetch (CPU job, network), every pipeline step (GPU job), upload for the Pi (CPU job). Steps skip
# finished work, so a stopped contest is simply started again. Afterwards the model's type copies are deleted
# unless a later contest here or a running one needs them. Log: data/logs/run.log
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${PANTHER_WORK:-/s3/panther/work}/data/logs/run.log"
R=(/opt/venv/bin/python scripts/server/pipeline.py)
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
job() {  # job cpu|gpu NAME CMD...: run, wait, 0 if it exited 0
  local kind="$1" name="$2"; shift 2
  "$HERE/panther.sh" "${kind/cpu/}job" "$name" "$@" >> "$LOG" 2>&1 || return 1
  [ "$("$HERE/panther.sh" wait "$name" | head -1)" = 0 ]
}
contests=("$@")
for i in "${!contests[@]}"; do
  read -r B M <<< "${contests[$i]}"
  n="$(echo "$M" | cut -d- -f2 | tr 'A-Z' 'a-z')-b$B-$(date +%s)"
  log "start $B GiB $M"
  if job cpu "fetch-$n" "${R[@]}" fetch "$M" && job gpu "run-$n" "${R[@]}" run "$B" "$M" \
     && job cpu "upload-$n" "${R[@]}" upload "$B" "$M"; then log "done $B GiB $M"; else log "FAILED $B GiB $M"; fi
  later="${contests[*]:$((i + 1))}"
  if ! grep -qw -- "$M" <<< "$later" && ! pgrep -f "pipeline.py run [0-9]+ $M" > /dev/null; then
    find "${PANTHER_WORK:-/s3/panther/work}/data/store/$M" -name "$M-*.gguf" ! -name "*-BF16.gguf" -delete
    log "deleted the type copies of $M"
  fi
done
