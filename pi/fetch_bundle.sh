#!/usr/bin/env bash
# Copy the Pi bundle from the server to the Pi, through this machine (Phase E2). Run on your Mac, with the
# server's SSH control master open (see deploy/server/local.env). Two hops: server -> here -> Pi.
#
#   pi/fetch_bundle.sh PI_HOST            # e.g. pi/fetch_bundle.sh pi@raspberrypi.local
set -euo pipefail
PI="${1:?usage: pi/fetch_bundle.sh PI_HOST}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/deploy/server/local.env"
LOCAL="${PANTHER_BUNDLE_CACHE:-$HOME/panther-bundle}"
W="${PANTHER_WORK:-/s3/panther/work}"
mkdir -p "$LOCAL"
rsync -av --progress -e "ssh -o ControlPath=$PANTHER_SSH_CONTROL -o BatchMode=yes" \
  "$PANTHER_HOST:$W/data/pi_bundle/" "$LOCAL/"
ssh "$PI" 'mkdir -p ~/panther-pi/bundle'
rsync -av --progress --exclude results/ "$LOCAL/" "$PI:panther-pi/bundle/"
echo "bundle copied to $PI:~/panther-pi/bundle"
