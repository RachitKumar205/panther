#!/usr/bin/env bash
# Run a chain of Panther jobs one after another on the server, unattended (D42). Each line of STEPS is
#   cpu|gpu  NAME  CMD...
# Each step starts as `panther.sh job|gpujob NAME CMD...`, the chain waits for it, and stops at the first
# step that exits non-zero. Steps whose container already exited 0 are skipped, so a chain can be rerun.
# Progress goes to $PANTHER_WORK/data/logs/chain-<CHAIN>.log. Start it detached:
#
#   setsid nohup deploy/server/chain.sh CHAIN STEPS_FILE >/dev/null 2>&1 &
set -uo pipefail
CHAIN="$1"; STEPS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${PANTHER_WORK:-/s3/panther/work}"
LOG="$W/data/logs/chain-$CHAIN.log"
P=(podman --root "$W/containers/storage" --runroot "/run/user/$(id -u)/panther" --events-backend file)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "chain $CHAIN start (CPUS=${PANTHER_CPUS:-20} MEM=${PANTHER_MEM:-64g})"
while read -r kind name cmd; do
  [ -z "$kind" ] || [ "${kind:0:1}" = "#" ] && continue
  status="$("${P[@]}" inspect -f '{{.State.Status}} {{.State.ExitCode}}' "panther-$name" 2>/dev/null || true)"
  if [ "$status" = "exited 0" ]; then log "skip $name (already done)"; continue; fi
  if [ "${status%% *}" = "running" ]; then
    log "adopt $name (already running)"  # a restructured chain picks up a job started by an earlier one
  else
    [ -n "$status" ] && "${P[@]}" rm -f "panther-$name" >/dev/null
    case "$kind" in cpu) sub=job ;; gpu) sub=gpujob ;; *) log "bad kind $kind"; exit 2 ;; esac
    # shellcheck disable=SC2086
    "$HERE/panther.sh" "$sub" "$name" $cmd >>"$LOG" 2>&1 || { log "FAILED to start $name"; exit 1; }
  fi
  rc="$("${P[@]}" wait "panther-$name")"
  log "$name exited $rc"
  [ "$rc" = 0 ] || { log "chain $CHAIN stopped at $name"; exit 1; }
done < "$STEPS"
log "chain $CHAIN done"
