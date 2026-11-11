#!/usr/bin/env bash
# Wait until the named chains have finished successfully, then run another chain (D42 final step).
#
#   setsid nohup deploy/server/after-chains.sh "7b 14b" final STEPS_FILE >/dev/null 2>&1 &
#
# Gives up (and runs nothing) if any awaited chain stops at a failure. Polls its log files every 2 minutes.
set -uo pipefail
WAIT_FOR="$1"; CHAIN="$2"; STEPS="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${PANTHER_WORK:-/s3/panther/work}/data/logs"
LOG="$LOGS/chain-$CHAIN.log"
echo "$(date -u +%FT%TZ) waiting for: $WAIT_FOR" >> "$LOG"
while :; do
  done_all=1
  for c in $WAIT_FOR; do
    if grep -q "chain $c stopped" "$LOGS/chain-$c.log" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) chain $c failed; $CHAIN not started" >> "$LOG"; exit 1
    fi
    grep -q "chain $c done" "$LOGS/chain-$c.log" 2>/dev/null || done_all=0
  done
  [ "$done_all" = 1 ] && break
  sleep 120
done
exec "$HERE/chain.sh" "$CHAIN" "$STEPS"
