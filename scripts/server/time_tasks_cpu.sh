#!/usr/bin/env bash
# D40 timing sample (replaces A4's full CPU stage, cut for time): CPU build, A1 file, 100 random selection
# questions per task (llama.cpp's own seeded subset). Timing only; no per-question results are recorded.
#
#   deploy/server/panther.sh job a4-time scripts/server/time_tasks_cpu.sh
set -euo pipefail
DATA="${PANTHER_DATA:?run inside the Panther image}"
B="$PANTHER_LLAMA/build-cpu/bin/llama-perplexity"
M="$DATA/gate/a1/a1.gguf"
T="$DATA/splits/llama_tasks/selection"
OUT="$DATA/gate/a4"
mkdir -p "$OUT"
N=100
: > "$OUT/cpu_timing.tsv"
for task in hellaswag arc_challenge mmlu winogrande; do
  if [ "$task" = winogrande ]; then mode=(-f "$T/$task.csv" --winogrande --winogrande-tasks "$N")
  else mode=(-bf "$T/$task.bin" --multiple-choice --multiple-choice-tasks "$N"); fi
  t0=$(date +%s.%N)
  "$B" -m "$M" "${mode[@]}" -c 4096 -np 5 -t "${PANTHER_THREADS:-16}" > "$OUT/cpu_timing_$task.log" 2>&1
  t1=$(date +%s.%N)
  grep -qE "Final (result|Winogrande)" "$OUT/cpu_timing_$task.log" || { echo "FAILED: $task" >&2; exit 1; }
  printf '%s\t%s\t%s\n' "$task" "$N" "$(python3 -c "print(round($t1-$t0,1))")" | tee -a "$OUT/cpu_timing.tsv"
done
