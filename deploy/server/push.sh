#!/usr/bin/env bash
# Copy the repo from this machine to the server (code only; no data). Uses the open SSH control master.
#
#   deploy/server/push.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${PANTHER_HOST:-svcuser@server.example}"
CP="${PANTHER_SSH_CONTROL:-$HOME/.ssh/cm-panther}"
DEST="${PANTHER_WORK:-/s3/panther/work}/repo/"

rsync -a --delete -e "ssh -o ControlPath=$CP -o BatchMode=yes" \
  --exclude .git/ --exclude .venv/ --exclude /data/ --exclude /third_party/ --exclude /results/ \
  --exclude __pycache__/ --exclude .pytest_cache/ --exclude '*.gguf' \
  "$ROOT/" "$HOST:$DEST"
echo "pushed to $HOST:$DEST"
