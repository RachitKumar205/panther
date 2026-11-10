#!/usr/bin/env bash
# Phase A gate, step A1: llama.cpp side (docs/PIPELINE.md). Run inside the Panther image:
#
#   deploy/server/panther.sh run    scripts/server/gate_a1.sh cpu    # assemble + CPU-build load/generate/ppl
#   deploy/server/panther.sh gpurun scripts/server/gate_a1.sh gpu    # CUDA-build load/generate/ppl x4
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/gate_a1.py report
#
# Perplexity uses the first 16 chunks x 2048 tokens of the *sensitivity* text (never test).
set -euo pipefail

STAGE="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${PANTHER_DATA:?run inside the Panther image}"
THREADS="${PANTHER_THREADS:-16}"
MODEL=Qwen2.5-1.5B-Instruct
STORE="$DATA/store/$MODEL"
OUT="$DATA/gate/a1"
TEXT="$DATA/splits/text/sensitivity.txt"
PROMPTS=(
  "Name three primary colours."
  "Write one sentence explaining what a CPU does."
  "What is 17 + 25? Answer with the number only."
)

ppl() {  # ppl BUILD NAME MODEL_FILE [extra args]
  local build="$1" name="$2" model="$3"; shift 3
  # -lv 4: at this llama.cpp version the loader prints per-type tensor counts only at verbosity >= 4.
  "$PANTHER_LLAMA/build-$build/bin/llama-perplexity" -m "$model" -f "$TEXT" -c 2048 -b 2048 --chunks 16 \
    -t "$THREADS" -lv 4 "$@" > "$OUT/ppl-$build-$name.log" 2>&1
  grep -E "Final estimate" "$OUT/ppl-$build-$name.log"
}

gen() {  # gen BUILD [extra args]
  local build="$1"; shift
  local i=0
  for p in "${PROMPTS[@]}"; do
    i=$((i + 1))
    "$PANTHER_LLAMA/build-$build/bin/llama-cli" -m "$OUT/a1.gguf" -st -p "$p" -n 64 --temp 0 --seed 1 \
      -c 2048 -t "$THREADS" "$@" < /dev/null 2>/dev/null \
      | sed -n '/^> /,/^\[ Prompt:/p' | sed '1d;$d' > "$OUT/gen-$build-$i.txt"
    echo "[$build $i] $(head -c 200 "$OUT/gen-$build-$i.txt" | tr '\n' ' ')"
  done
}

case "$STAGE" in
  cpu)
    "$PANTHER_PY" "$ROOT/scripts/server/gate_a1.py" assemble
    ppl cpu mixed "$OUT/a1.gguf"
    gen cpu
    ;;
  gpu)
    [ -f "$OUT/a1.gguf" ] || { echo "run the cpu stage first" >&2; exit 1; }
    ppl cuda mixed "$OUT/a1.gguf" -ngl 99
    ppl cuda q2_k "$STORE/$MODEL-Q2_K.gguf" -ngl 99
    ppl cuda q8_0 "$STORE/$MODEL-Q8_0.gguf" -ngl 99
    ppl cuda bf16 "$STORE/$MODEL-BF16.gguf" -ngl 99
    gen cuda -ngl 99
    ;;
  *) echo "usage: gate_a1.sh cpu|gpu" >&2; exit 2 ;;
esac
