#!/usr/bin/env bash
# Phase A gate, step A2: llama.cpp side (docs/PIPELINE.md). Run inside the Panther image, in order:
#
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/gate_a2.py prep
#   deploy/server/panther.sh gpujob a2 scripts/server/gate_a2.sh gpu      # references, self floor, study KL
#   deploy/server/panther.sh run    scripts/server/gate_a2.sh cpu         # CPU-build floor
#   deploy/server/panther.sh run    $PANTHER_PY scripts/server/gate_a2.py analyze
#
# Every KL run repeats the base run's -c/-b (D28).
set -euo pipefail

STAGE="$1"
DATA="${PANTHER_DATA:?run inside the Panther image}"
THREADS="${PANTHER_THREADS:-16}"
MODEL=Qwen2.5-1.5B-Instruct
BF16="$DATA/store/$MODEL/$MODEL-BF16.gguf"
REF="$DATA/reference/$MODEL"
OUT="$DATA/gate/a2"
STUDY="$OUT/study"
CB=(-c 2048 -b 2048 -t "$THREADS")

# llama-perplexity exits 0 even on errors, so check what each run produced.
need_ref() { [ -s "$1" ] || { echo "FAILED: no reference written ($1); see $2" >&2; tail -3 "$2" >&2; exit 1; }; }
need_kl()  { grep -q "Mean    KLD" "$1" || { echo "FAILED: no KL summary in $1" >&2; tail -3 "$1" >&2; exit 1; }; }
mkdir -p "$REF" "$OUT" "$STUDY/refs" "$STUDY/logs"

case "$STAGE" in
  gpu)
    B="$PANTHER_LLAMA/build-cuda/bin"
    for split in sensitivity selection; do
      [ -f "$REF/$split.kld" ] && continue
      "$B/llama-perplexity" -m "$BF16" -f "$DATA/splits/text/$split.txt" "${CB[@]}" -ngl 99 \
        --kl-divergence-base "$REF/$split.kld.part" > "$OUT/ref-$split.log" 2>&1
      need_ref "$REF/$split.kld.part" "$OUT/ref-$split.log"
      mv "$REF/$split.kld.part" "$REF/$split.kld"
      grep "Final estimate" "$OUT/ref-$split.log"
    done
    "$B/llama-perplexity" -m "$BF16" --kl-divergence-base "$REF/sensitivity.kld" --kl-divergence \
      "${CB[@]}" -ngl 99 > "$OUT/floor-self.log" 2>&1
    need_kl "$OUT/floor-self.log"
    echo "storage floor done"

    # Chunk-count study: one reference and one KL run per chunk, for exact per-chunk means.
    for k in $(seq -w 0 47); do
      [ -f "$STUDY/refs/chunk-$k.kld" ] && continue
      "$B/llama-perplexity" -m "$BF16" -f "$STUDY/chunks/chunk-$k.txt" "${CB[@]}" -ngl 99 --chunks 1 \
        --kl-divergence-base "$STUDY/refs/chunk-$k.kld.part" > "$STUDY/logs/ref-$k.log" 2>&1
      need_ref "$STUDY/refs/chunk-$k.kld.part" "$STUDY/logs/ref-$k.log"
      mv "$STUDY/refs/chunk-$k.kld.part" "$STUDY/refs/chunk-$k.kld"
    done
    echo "per-chunk references done"
    for m in "$STUDY"/models/*.gguf; do
      name="$(basename "$m" .gguf)"
      for k in $(seq -w 0 47); do
        "$B/llama-perplexity" -m "$m" --kl-divergence-base "$STUDY/refs/chunk-$k.kld" --kl-divergence \
          "${CB[@]}" -ngl 99 > "$STUDY/logs/$name-$k.log" 2>&1
        need_kl "$STUDY/logs/$name-$k.log"
      done
      echo "study $name done"
    done
    "$B/llama-perplexity" -m "$STUDY/models/base.gguf" --kl-divergence-base "$REF/sensitivity.kld" \
      --kl-divergence "${CB[@]}" -ngl 99 > "$OUT/study-base-contiguous.log" 2>&1
    need_kl "$OUT/study-base-contiguous.log"
    echo "gpu stage done"
    ;;
  cpu)
    "$PANTHER_LLAMA/build-cpu/bin/llama-perplexity" -m "$BF16" --kl-divergence-base "$REF/sensitivity.kld" \
      --kl-divergence "${CB[@]}" > "$OUT/floor-cpu.log" 2>&1
    need_kl "$OUT/floor-cpu.log"
    echo "cpu floor done"
    ;;
  *) echo "usage: gate_a2.sh gpu|cpu" >&2; exit 2 ;;
esac
