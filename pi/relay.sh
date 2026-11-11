#!/usr/bin/env bash
# Run on the Mac: wait until a model's winner is bundled on the server, copy it to the Pi, and run the Pi check
# (load once, peak RAM vs budget, one speed run). The server and the Pi cannot reach each other; the Mac relays.
#
#   pi/relay.sh 7 14b [PI_HOST]
set -euo pipefail
B="${1:?budget}"; S="${2:?model size, e.g. 14b}"; PI="${3:-jia@100.89.94.89}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/deploy/server/wait-job.sh" "rw-bundle$B-$S" 600 | tail -2
"$ROOT/pi/fetch_bundle.sh" "$PI" full "budget-$B-$S" | tail -1
ssh -o BatchMode=yes "$PI" "cd ~/panther && git pull -q && python3 pi/panther_pi.py run --quick --bundle ~/panther-pi/budget-$B-$S | tail -15"
