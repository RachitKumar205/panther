#!/usr/bin/env bash
# Phase A gate, step A4 (docs/PIPELINE.md). Inside the Panther image, in order:
#
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/build_tasks.py     # convert + verify
#   deploy/server/panther.sh gpujob a4-gpu scripts/server/gate_a4.sh gpu         # 4 files x 4 tasks + D39
#   deploy/server/panther.sh job    a4-cpu scripts/server/gate_a4.sh cpu         # BF16, A1 on the CPU build
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/gate_a4.py
#
# Selection items only (D34).
set -euo pipefail
STAGE="$1"
DATA="${PANTHER_DATA:?run inside the Panther image}"
MODEL=Qwen2.5-1.5B-Instruct
STORE="$DATA/store/$MODEL"
TASKS=(hellaswag arc_challenge mmlu winogrande)

run_tasks() {  # run_tasks BUILD LABEL GGUF
  for t in "${TASKS[@]}"; do
    "$PANTHER_PY" -m panther.quality.run task --model "$MODEL" --build "$1" --gguf "$3" \
      --task "$t" --split selection --label "$2"
  done
}

case "$STAGE" in
  gpu)
    run_tasks cuda a4:bf16 "$STORE/$MODEL-BF16.gguf"
    run_tasks cuda a4:q8_0 "$STORE/$MODEL-Q8_0.gguf"
    run_tasks cuda a4:q4_k "$STORE/$MODEL-Q4_K.gguf"
    run_tasks cuda a4:a1   "$DATA/gate/a1/a1.gguf"
    "$PANTHER_PY" "$(dirname "$0")/check_d39.py"
    ;;
  cpu)
    run_tasks cpu a4:a1   "$DATA/gate/a1/a1.gguf"
    run_tasks cpu a4:bf16 "$STORE/$MODEL-BF16.gguf"
    ;;
  *) echo "usage: gate_a4.sh gpu|cpu" >&2; exit 2 ;;
esac
