#!/usr/bin/env bash
# Run on the Mac: keep one Pi relay alive per pending winner. Every 5 minutes, for each "BUDGET SIZE" given, start
# pi/relay.sh if none is running and the Pi has no result for that bundle yet. Only while the server's SSH control
# master is up: it never opens a new SSH login (the server penalises repeated failed logins).
#
#   pi/supervise.sh "5 7b" "5 14b" "7 20b"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/deploy/server/local.env"
PI="${PANTHER_PI:-jia@100.89.94.89}"
mkdir -p "$HOME/panther-logs"
while :; do
  if ssh -o ControlPath="$PANTHER_SSH_CONTROL" -O check "$PANTHER_HOST" 2>/dev/null; then
    for item in "$@"; do
      read -r B S <<< "$item"
      pgrep -f "[r]elay.sh $B $S " >/dev/null && continue
      ssh -o BatchMode=yes -o ConnectTimeout=20 "$PI" "grep -q 'peak RAM' ~/panther-pi/budget-$B-$S/results/queue.log 2>/dev/null" && continue
      echo "$(date +%H:%M) starting relay $B $S"
      nohup "$ROOT/pi/relay.sh" "$B" "$S" "$PI" >> "$HOME/panther-logs/pi-$B-$S.log" 2>&1 &
    done
  fi
  sleep 300
done
