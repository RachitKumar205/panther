#!/usr/bin/env bash
# Pause Panther's containers if the co-located service degrades; resume when it recovers.
#
#   setsid nohup deploy/server/guard.sh >/dev/null 2>&1 &
#
# Bad check: not HTTP 200 or slower than 3 s. 3 bad checks in a row -> `podman pause` every running panther-*
# container. 5 good checks in a row while paused -> `podman unpause`. Checks every 60 s. Touches nothing else.
set -uo pipefail
PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"
URL="${PANTHER_HEALTH_URL:?set PANTHER_HEALTH_URL in deploy/server/local.env}"
W="${PANTHER_WORK:-/s3/panther/work}"
LOG="$W/data/logs/guard.log"
P=(podman --root "$W/containers/storage" --runroot "/run/user/$(id -u)/panther" --events-backend file)
bad=0; good=0; paused=0
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "guard start ($URL)"
while :; do
  out="$(curl -sL -o /dev/null -w '%{http_code} %{time_total}' --max-time 10 "$URL" 2>/dev/null || echo "000 10")"
  code="${out%% *}"; t="${out##* }"
  if [ "$code" = 200 ] && awk -v t="$t" 'BEGIN{exit !(t < 3.0)}'; then good=$((good+1)); bad=0; else bad=$((bad+1)); good=0; log "bad check: $out"; fi
  if [ "$paused" = 0 ] && [ "$bad" -ge 3 ]; then
    names="$("${P[@]}" ps --filter status=running --format '{{.Names}}' | grep '^panther-' || true)"
    [ -n "$names" ] && "${P[@]}" pause $names >/dev/null 2>&1
    paused=1; log "PAUSED: $names"
  elif [ "$paused" = 1 ] && [ "$good" -ge 5 ]; then
    names="$("${P[@]}" ps --filter status=paused --format '{{.Names}}' | grep '^panther-' || true)"
    [ -n "$names" ] && "${P[@]}" unpause $names >/dev/null 2>&1
    paused=0; log "resumed: $names"
  fi
  sleep 60
done
