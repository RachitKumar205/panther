#!/usr/bin/env bash
# Wait (from the Mac) for a Panther job on the server to finish, then print its exit code and log tail.
#
#   deploy/server/wait-job.sh NAME [max-minutes]
#
# Stops at the FIRST SSH failure (exit 255: connection expired, refused, or unreachable) instead of
# retrying: the server's sshd penalises repeated failed logins from one address (OpenSSH PerSourcePenalties,
# banner "Not allowed at this time"), and a retry loop locks the user out (D30). Polls every 30 s.
set -uo pipefail
# Site settings (host, user, health-check URL) live in the git-ignored deploy/server/local.env.
PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"

NAME="$1"
MAX_MIN="${2:-120}"
HOST="${PANTHER_HOST:?set PANTHER_HOST in deploy/server/local.env}"
CP="${PANTHER_SSH_CONTROL:?set PANTHER_SSH_CONTROL in deploy/server/local.env}"
SSH=(ssh -o "ControlPath=$CP" -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
PODMAN='podman --root /s3/panther/work/containers/storage --runroot /run/user/$(id -u)/panther --events-backend file'

# Never even try without a live control master: a failed attempt counts against us.
ssh -o "ControlPath=$CP" -O check "$HOST" 2>/dev/null || { echo "no SSH control master; reopen it first" >&2; exit 3; }

deadline=$(( $(date +%s) + MAX_MIN * 60 ))
while :; do
  s="$("${SSH[@]}" "$PODMAN inspect -f '{{.State.Status}} {{.State.ExitCode}}' panther-$NAME")"
  rc=$?
  if [ "$rc" = 255 ]; then echo "SSH failed; stopping without retrying (reopen the control master)" >&2; exit 3; fi
  case "$s" in exited*) break ;; esac
  [ "$(date +%s)" -lt "$deadline" ] || { echo "still running after $MAX_MIN min: $s"; exit 4; }
  sleep 30
done
echo "panther-$NAME: $s"
"${SSH[@]}" "tail -20 /s3/panther/work/data/logs/$NAME.log; curl -sL -o /dev/null -w 'health %{http_code}\n' $PANTHER_HEALTH_URL"
