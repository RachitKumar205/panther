#!/usr/bin/env bash
# D35: KL on the sensitivity text for all 12 uniform 1.5B formats on the CPU build, to compare with the
# A3 CUDA sweep. Inside the Panther image:
#
#   deploy/server/panther.sh job d35 scripts/server/check_d35.sh
#   deploy/server/panther.sh run $PANTHER_PY scripts/server/check_d35.py
set -euo pipefail
DATA="${PANTHER_DATA:?run inside the Panther image}"
MODEL=Qwen2.5-1.5B-Instruct
for f in Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0 Q4_0 IQ2_XS IQ3_XXS IQ3_S IQ4_XS IQ4_NL; do
  "$PANTHER_PY" -m panther.quality.run kl --model "$MODEL" --build cpu \
    --gguf "$DATA/store/$MODEL/$MODEL-$f.gguf" --reference sensitivity --label "d35:uniform:$f"
done
