#!/usr/bin/env bash
# Build the per-format tensor store for one model . Runs inside the Panther image:
#
#   deploy/server/panther.sh gpujob store-1.5b scripts/server/build_tensor_store.sh Qwen2.5-1.5B-Instruct <calibration.txt>
#
# PANTHER_FORMATS="Q8_0 Q6_K ..." limits the formats built (proof of concept, D42); default: all candidates.
#
# Steps (weights fetched beforehand by fetch_model.sh): convert to BF16 GGUF -> importance matrix on the
# calibration split -> one uniform GGUF per candidate format, with --pure and explicit embedding and
# output types so every searched tensor holds exactly that format. Mixed files are then assembled with
# panther.gguf_files. Each step is skipped if its output already exists.
set -euo pipefail

MODEL="$1"
CALIB="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="${PANTHER_LLAMA:?run inside the Panther image}"
DATA="${PANTHER_DATA:-$ROOT/data}"
PY="${PANTHER_PY:?}"
THREADS="${PANTHER_THREADS:-16}"
BIN="$LLAMA/build-cuda/bin"
OUT="$DATA/store/$MODEL"
mkdir -p "$OUT"
TIMINGS="$OUT/.timings"   # "step seconds" lines, read by store_manifest.py
timed() { local step="$1" t0=$SECONDS; shift; "$@"; echo "$step $((SECONDS - t0))" >> "$TIMINGS"; }

read -r REPO REV < <("$PY" - "$MODEL" <<'EOF'
import sys
from panther.model import load_registry
m = next(m for m in load_registry()["models"] if m["name"] == sys.argv[1])
print(m["repo"], m["revision"])
EOF
)

# GPU jobs have no network: the weights must already be fetched (scripts/server/fetch_model.sh, CPU job).
HF_DIR="$DATA/hf/$MODEL"
if [ "$(cat "$HF_DIR/.panther-revision" 2>/dev/null)" != "$REPO@$REV" ]; then
  echo "weights for $REPO@$REV not fetched; run scripts/server/fetch_model.sh $MODEL as a CPU job first" >&2
  exit 1
fi

BF16="$OUT/$MODEL-BF16.gguf"
if [ ! -f "$BF16" ]; then
  # InternLM's tokenizer needs protobuf 7.36 (its generated code); the image has 4.25. A newer one sits in data/pydeps-convert
  # (uv pip install --target, inside the container), used by the converter only.
  PYTHONPATH="$DATA/pydeps-convert${PYTHONPATH:+:$PYTHONPATH}" timed convert "$PANTHER_CONVERT_PY" "$LLAMA/convert_hf_to_gguf.py" "$HF_DIR" --outtype bf16 --outfile "$BF16.part"
  mv "$BF16.part" "$BF16"
  rm -rf "$HF_DIR"   # the 16-bit GGUF is the original from here on (docs/STANDARD.md)
fi

IMAT="$OUT/$MODEL.imatrix.gguf"
if [ ! -f "$IMAT" ]; then
  PO="$("$PY" -c "import sys; from panther.model import head_has_imatrix; print('--process-output' if head_has_imatrix(sys.argv[1]) else '')" "$MODEL")"
  timed imatrix "$BIN/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IMAT.part.gguf" -c 512 -ngl 99 -t "$THREADS" $PO
  mv "$IMAT.part.gguf" "$IMAT"
fi

# Candidate format -> llama-quantize ftype name. With --pure, the *_S mixes are all one type.
declare -A FTYPE=(
  [Q2_K]=Q2_K [Q3_K]=Q3_K_S [Q4_K]=Q4_K_S [Q5_K]=Q5_K_S [Q6_K]=Q6_K [Q8_0]=Q8_0 [Q4_0]=Q4_0
  [IQ1_S]=IQ1_S [IQ1_M]=IQ1_M [IQ2_XXS]=IQ2_XXS [IQ2_S]=IQ2_M  # the IQ2_S ftype's base type is IQ2_XS; IQ2_M's is IQ2_S
  [IQ2_XS]=IQ2_XS [IQ3_XXS]=IQ3_XXS [IQ3_S]=IQ3_S [IQ4_XS]=IQ4_XS [IQ4_NL]=IQ4_NL
)
# One line per format: "<format> <token-embedding format> <output-head format>". The embedding has no imatrix data, so
# formats that require it (configs/formats.yaml) put a placeholder there; the search never uses it.
PLAN=$("$PY" - "$MODEL" <<'EOF'
import sys
from panther.model import build_inventory, candidate_formats, load_geometry, source_assignment
inv = build_inventory(load_geometry(sys.argv[1]))
embd = next(g.gene_id for g in inv.genes if g.kind in ("embd", "embd_head"))
head = next((g.gene_id for g in inv.genes if g.kind == "head"), embd)  # tied models: the same gene
import os
wanted = os.environ.get("PANTHER_FORMATS", "").split() or candidate_formats()  # PoC: per-model menus (D42)
for f in wanted:
    sa = source_assignment(inv, f)
    print(f, sa[embd], sa[head])
EOF
)

while read -r F E H; do
  DST="$OUT/$MODEL-$F.gguf"
  [ -f "$DST" ] && continue
  T="$(echo "$F" | tr 'A-Z' 'a-z')"
  TE="$(echo "$E" | tr 'A-Z' 'a-z')"
  TH="$(echo "$H" | tr 'A-Z' 'a-z')"
  timed "quantize_$F" "$BIN/llama-quantize" --pure --imatrix "$IMAT" \
    --token-embedding-type "$TE" --output-tensor-type "$TH" \
    "$BF16" "$DST.part" "${FTYPE[$F]}" "$THREADS"
  mv "$DST.part" "$DST"
done <<< "$PLAN"

# Verify every source (every searched tensor holds exactly its format, no silent fallback) and write
# store/<model>/manifest.json with inputs, hashes, verification and timings. Fails on any mismatch.
"$PY" "$ROOT/scripts/server/store_manifest.py" "$MODEL" "$CALIB" --timings "$TIMINGS" ${PANTHER_FORMATS:+--formats $PANTHER_FORMATS}
