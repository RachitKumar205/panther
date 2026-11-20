#!/usr/bin/env bash
# Sets up the Raspberry Pi 5: llama.cpp's server at the pinned commit (~/panther-pi/llama.cpp) and the Hugging Face
# CLI (~/hf-venv, huggingface_hub at the version in uv.lock). Run as an ordinary user, not root.
# Prerequisite, run once by the Pi's admin (the only step that needs sudo):
#   sudo apt-get install -y git cmake build-essential python3 python3-venv
#
#   bash pi/setup.sh
set -euo pipefail
[ "$(id -u)" != 0 ] || { echo "run as an ordinary user, not root" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$HOME/panther-pi/llama.cpp"
HUB="$(awk '/^name = "huggingface-hub"$/{f=1; next} f&&/^version = /{gsub(/"/, "", $3); print $3; exit}' "$ROOT/uv.lock")"
[ -n "$HUB" ] || { echo "no huggingface-hub version in uv.lock" >&2; exit 1; }
COMMIT="$(awk '/^llama_cpp:/{f=1} f&&/commit:/{print $2; exit}' "$ROOT/configs/models.yaml")"
for t in git cmake g++ python3; do command -v "$t" >/dev/null || { echo "missing $t (see the apt line above)" >&2; exit 1; }; done
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
cmake --build "$LLAMA/build" -j 4 --target llama-server
"$LLAMA/build/bin/llama-server" --version 2>&1 | head -2
echo "llama.cpp $COMMIT built in $LLAMA/build/bin"
[ -x "$HOME/hf-venv/bin/python" ] || python3 -m venv "$HOME/hf-venv"
"$HOME/hf-venv/bin/pip" install -q "huggingface_hub==$HUB"
"$HOME/hf-venv/bin/hf" --help >/dev/null
echo "huggingface_hub $HUB installed in ~/hf-venv. One manual step left (a write token, to upload results):"
echo "  ~/hf-venv/bin/hf auth login"
