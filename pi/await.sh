#!/usr/bin/env bash
# Run on the Pi: wait for each winner to appear in its private Hugging Face repo, download it and run the Pi check,
# one at a time. Needs neither the Mac nor the server: the server uploads (pipeline.py upload), the Pi polls.
#
#   nohup bash pi/await.sh therachitkumar 5:7b 5:14b 7:20b 7:24b 5:20b > ~/panther-pi/await.log 2>&1 &
USER_NS="${1:?hf user}"; shift
cd "$(dirname "$0")/.." || exit 1
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for item in "${pending[@]}"; do
    B="${item%%:*}"; S="${item##*:}"; D="$HOME/panther-pi/budget-$B-$S"
    if grep -q 'peak RAM' "$D/results/queue.log" 2>/dev/null; then echo "$(date +%H:%M) $B GiB $S already checked"; continue; fi
    if flock /tmp/panther-check.lock ~/hf-venv/bin/hf download "$USER_NS/panther-$S-${B}g" --local-dir "$D" --quiet \
         >/dev/null 2>&1 && [ -f "$D/manifest.json" ] && ls "$D"/*.gguf >/dev/null 2>&1; then
      echo "$(date +%H:%M) $B GiB $S downloaded; checking"
      git pull -q
      flock /tmp/panther-check.lock python3 pi/panther_pi.py run --quick --bundle "$D" | tail -4
    else
      left+=("$item")
    fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 300
done
echo "$(date +%H:%M) all checked"
