#!/usr/bin/env bash
# Copy the repo from this machine to the server (code only; no data). Uses the open SSH control master.
#
#   deploy/server/push.sh
set -euo pipefail
# Site settings (host, user, health-check URL) live in the git-ignored deploy/server/local.env.
PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${PANTHER_HOST:?set PANTHER_HOST in deploy/server/local.env}"
CP="${PANTHER_SSH_CONTROL:?set PANTHER_SSH_CONTROL in deploy/server/local.env}"
DEST="${PANTHER_WORK:-/s3/panther/work}/repo/"

rsync -a --delete -e "ssh -o ControlPath=$CP -o BatchMode=yes" \
  --exclude .git/ --exclude .venv/ --exclude /data/ --exclude /third_party/ --exclude /results/ \
  --exclude __pycache__/ --exclude .pytest_cache/ --exclude '*.gguf' \
  "$ROOT/" "$HOST:$DEST"
echo "pushed to $HOST:$DEST"
