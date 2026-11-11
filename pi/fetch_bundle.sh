#!/usr/bin/env bash
# Copy the Pi bundle from the server to the Pi, through this machine (Phase E2). Run on your Mac, with the
# server's SSH control master open (see deploy/server/local.env). Two hops: server -> here -> Pi.
#
#   pi/fetch_bundle.sh PI_HOST            # e.g. pi/fetch_bundle.sh pi@raspberrypi.local (all files, ~25 GB)
#   pi/fetch_bundle.sh PI_HOST quick      # manifest + agreement text + recommended file only (~6 GB, D44)
set -euo pipefail
PI="${1:?usage: pi/fetch_bundle.sh PI_HOST [quick]}"
MODE="${2:-full}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/deploy/server/local.env"
LOCAL="${PANTHER_BUNDLE_CACHE:-$HOME/panther-bundle}"
W="${PANTHER_WORK:-/s3/panther/work}"
mkdir -p "$LOCAL"
SSH_S="ssh -o ControlPath=$PANTHER_SSH_CONTROL -o BatchMode=yes"
if [ "$MODE" = quick ]; then
  rsync -av -e "$SSH_S" "$PANTHER_HOST:$W/data/pi_bundle/manifest.json" "$PANTHER_HOST:$W/data/pi_bundle/agreement.txt" "$LOCAL/"
  REC="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_file"])' "$LOCAL/manifest.json")"
  rsync -av --progress -e "$SSH_S" "$PANTHER_HOST:$W/data/pi_bundle/$REC" "$LOCAL/"
  INCLUDE=(--include manifest.json --include agreement.txt --include "$REC" --exclude '*')
else
  rsync -av --progress -e "$SSH_S" "$PANTHER_HOST:$W/data/pi_bundle/" "$LOCAL/"
  INCLUDE=(--exclude results/)
fi
ssh "$PI" 'mkdir -p ~/panther-pi/bundle'
rsync -av --progress "${INCLUDE[@]}" "$LOCAL/" "$PI:panther-pi/bundle/"
echo "bundle copied to $PI:~/panther-pi/bundle"
