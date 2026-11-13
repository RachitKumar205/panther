#!/usr/bin/env bash
# Run on the Pi: wait for each winner to appear in its private Hugging Face repo, download it, check it
# (panther_pi.py), upload the result line to <user>/panther-status (the status page shows it), one at a time.
#
#   nohup bash pi/await.sh <hf user> 5:14b 7:24b > ~/panther-pi/await.log 2>&1 &
USER_NS="${1:?hf user}"; shift
SUF="${PANTHER_SUFFIX--v2}"
cd "$(dirname "$0")/.." || exit 1
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for item in "${pending[@]}"; do
    B="${item%%:*}"; S="${item##*:}"; D="$HOME/panther-pi/budget-$B-$S$SUF"
    grep -q 'peak RAM' "$D/results/queue.log" 2>/dev/null && continue
    if ~/hf-venv/bin/hf download "$USER_NS/panther-$S-${B}g$SUF" --local-dir "$D" --quiet >/dev/null 2>&1 \
       && ls "$D"/*.gguf >/dev/null 2>&1; then
      git pull -q
      python3 pi/panther_pi.py run --bundle "$D" && grep "peak RAM" "$D/results/queue.log" | tail -1 \
        > "$D/results/pi-result.txt" && ~/hf-venv/bin/hf upload "$USER_NS/panther-status" \
        "$D/results/pi-result.txt" "pi/$B-$S$SUF.txt" --commit-message "Pi $B GiB $S" >/dev/null 2>&1
      rm -f "$D"/*.gguf  # the SD card holds one model at a time
    else
      left+=("$item")
    fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 300
done
