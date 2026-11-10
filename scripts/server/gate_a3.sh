#!/usr/bin/env bash
# Phase A gate, step A3: validate the evaluator on 1.5B (docs/PIPELINE.md). Inside the Panther image:
#
#   deploy/server/panther.sh gpujob a3-gpu scripts/server/gate_a3.sh gpu
#   deploy/server/panther.sh job    a3-cpu scripts/server/gate_a3.sh cpu
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/gate_a3.py
#
# Sensitivity and selection text only; never test (D34).
set -euo pipefail

STAGE="$1"
DATA="${PANTHER_DATA:?run inside the Panther image}"
MODEL=Qwen2.5-1.5B-Instruct
STORE="$DATA/store/$MODEL"
A1="$DATA/gate/a1/a1.gguf"
EV=("$PANTHER_PY" -m panther.quality.run)

case "$STAGE" in
  gpu)
    B=(--model "$MODEL" --build cuda)
    "${EV[@]}" kl  "${B[@]}" --gguf "$A1" --reference sensitivity --label a3:a1
    "${EV[@]}" kl  "${B[@]}" --gguf "$A1" --reference selection   --label a3:a1
    for f in Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0 Q4_0 IQ2_XS IQ3_XXS IQ3_S IQ4_XS IQ4_NL; do
      "${EV[@]}" kl "${B[@]}" --gguf "$STORE/$MODEL-$f.gguf" --reference sensitivity --label "a3:uniform:$f"
    done
    "${EV[@]}" ppl "${B[@]}" --gguf "$A1" --text selection --label a3:a1
    "${EV[@]}" ppl "${B[@]}" --gguf "$STORE/$MODEL-BF16.gguf" --text selection --label a3:bf16
    "${EV[@]}" kl  "${B[@]}" --gguf "$A1" --reference sensitivity --label a3:a1:repeat
    ;;
  cpu)
    B=(--model "$MODEL" --build cpu)
    "${EV[@]}" kl  "${B[@]}" --gguf "$A1" --reference sensitivity --label a3:a1
    "${EV[@]}" kl  "${B[@]}" --gguf "$A1" --reference selection   --label a3:a1
    "${EV[@]}" ppl "${B[@]}" --gguf "$A1" --text selection --label a3:a1
    "${EV[@]}" ppl "${B[@]}" --gguf "$STORE/$MODEL-BF16.gguf" --text selection --label a3:bf16
    ;;
  *) echo "usage: gate_a3.sh gpu|cpu" >&2; exit 2 ;;
esac
