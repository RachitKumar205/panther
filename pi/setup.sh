#!/usr/bin/env bash
# Build llama.cpp on the Raspberry Pi 5 at the pinned commit (Phase E1). Run as an ordinary user, not root.
# Prerequisite, run once by the Pi's admin (the only step that needs sudo):
#   sudo apt-get install -y git cmake build-essential python3
#
#   bash pi/setup.sh            # builds into ~/panther-pi/llama.cpp
set -euo pipefail
[ "$(id -u)" != 0 ] || { echo "run as an ordinary user, not root" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${PANTHER_PI_DIR:-$HOME/panther-pi}"
LLAMA="$BASE/llama.cpp"
COMMIT="$(awk '/^llama_cpp:/{f=1} f&&/commit:/{print $2; exit}' "$ROOT/configs/models.yaml")"
for t in git cmake g++ python3; do command -v "$t" >/dev/null || { echo "missing $t (see the apt line above)" >&2; exit 1; }; done
mkdir -p "$BASE"
if [ ! -d "$LLAMA/.git" ]; then
  git init -q "$LLAMA"
  git -C "$LLAMA" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$LLAMA" fetch -q --depth 1 origin "$COMMIT"
git -C "$LLAMA" checkout -q FETCH_HEAD
[ "$(git -C "$LLAMA" rev-parse HEAD)" = "$COMMIT" ] || { echo "commit mismatch" >&2; exit 1; }
# CPU only, tuned for this CPU (Cortex-A76: dotprod, fp16); Release build.
cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$LLAMA/build" -j 4 --target llama-server llama-perplexity llama-cli llama-bench
"$LLAMA/build/bin/llama-server" --version 2>&1 | head -2
echo "llama.cpp $COMMIT built in $LLAMA/build/bin"
