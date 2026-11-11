#!/usr/bin/env bash
# Run on the Mac: wait until a model's Pi bundle is on its private Hugging Face repo, have the Pi download it,
# and run the Pi check (load once, peak RAM vs budget, one speed run). The server uploads (pipeline.py upload);
# the Pi signs in once with `~/hf-venv/bin/hf auth login`.
#
#   pi/relay.sh 7 14b [PI_HOST]
set -euo pipefail
B="${1:?budget}"; S="${2:?model size, e.g. 14b}"; PI="${3:-jia@100.89.94.89}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/deploy/server/local.env"
REPO="$(echo "$PANTHER_HF_REPO" | sed "s/{size}/$S/; s/{budget}/$B/")"
"$ROOT/deploy/server/wait-job.sh" "rw-upload$B-$S" 600 | tail -2
ssh -o BatchMode=yes "$PI" "~/hf-venv/bin/hf download $REPO --local-dir ~/panther-pi/budget-$B-$S --quiet"
echo "$(date +%H:%M) on the Pi: ~/panther-pi/budget-$B-$S (from $REPO)"
ssh -o BatchMode=yes "$PI" "cd ~/panther && git pull -q && python3 pi/panther_pi.py run --quick --bundle ~/panther-pi/budget-$B-$S | tail -15"
