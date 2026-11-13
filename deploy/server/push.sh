#!/usr/bin/env bash
# Copy the code (no data) to the server over the open SSH control master. Site details: git-ignored local.env.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/local.env"
rsync -a --delete -e "ssh -o ControlPath=${PANTHER_SSH_CONTROL:?} -o ProxyCommand=false -o BatchMode=yes" \
  --exclude .git/ --exclude .venv/ --exclude /data/ --exclude /third_party/ --exclude __pycache__/ \
  --exclude .pytest_cache/ --exclude '*.gguf' "$(cd "$(dirname "$0")/../.." && pwd)/" \
  "${PANTHER_HOST:?}:${PANTHER_WORK:-/s3/panther/work}/repo/"
