#!/usr/bin/env bash
# A small draft model for speculative decoding on the Pi: Qwen2.5-0.5B-Instruct (Apache-2.0; the same tokenizer as
# the Qwen2.5 contest models), converted and quantized with the image's pinned llama.cpp. Run in a CPU container:
#   ./panther cexec bash scripts/server/draft_model.sh
set -euo pipefail
M=Qwen2.5-0.5B-Instruct; H=/work/data/hf/$M; D=/work/data/draft
mkdir -p "$H" "$D"
"$PANTHER_HF" download "Qwen/$M" --local-dir "$H" --quiet
[ -f "$D/$M-BF16.gguf" ] || "$PANTHER_CONVERT_PY" "$PANTHER_LLAMA/convert_hf_to_gguf.py" "$H" --outtype bf16 --outfile "$D/$M-BF16.gguf"
for t in Q8_0 Q4_K_M; do
  [ -f "$D/$M-$t.gguf" ] || "$PANTHER_LLAMA/build-cpu/bin/llama-quantize" "$D/$M-BF16.gguf" "$D/$M-$t.gguf" "$t" >/dev/null
done
ls -la "$D" && sha256sum "$D"/*-Q*.gguf
