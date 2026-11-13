# scripts/server

`pipeline.py` runs one contest (a model at a RAM budget) inside the server's containers; the project README's "How
it works" explains every step. Each step skips work already on disk and writes progress lines to
`data/logs/progress-b<BUDGET>.log`.

| Step | Output |
|---|---|
| `fetch` (CPU container) | `data/hf/<model>` |
| `store` | `data/store/<model>/`: 16-bit original, importance matrix, one type copy per type (14), error-compensated |
| `error` | `rewrite/<model>/error.json` |
| `refs` | `data/reference/<model>/{selection,answers}.kld`, the original's answers and question results |
| `preset` | `rewrite/budget-B/preset-<model>.json` |
| `ga` | `rewrite/budget-B/list-<model>-v2.json` (top 3), `ga-<model>-v2.json` (every genome), `ga-<model>-live.json` |
| `final` | `rewrite/<model>/tuned/` (the distilled top 3), `rewrite/budget-B/winner-<model>-v2.json` (winner, verdict, held-out numbers for ours, preset, original) |
| `bundle`, `upload` | `data/pi_bundle/budget-B-<size>-v2/` → private Hugging Face repo `panther-<size>-<B>g-v2` |

`./panther start` (repo root) runs every contest through these steps; `pipeline.py run B MODEL` runs store → bundle
in order inside one GPU container. `status.py` rewrites `data/status.md` every 30 s
(verdicts, every contest's steps with time spent and left, the Pi's results) and publishes it to
`<user>/panther-status` every 60 s.

## In depth: every number and formula

### The RAM model (`src/panther/model.py`)

RAM(genome) = F32 norms and biases + Σ over matrices of bytes(matrix, its type) + KV cache + overhead, where KV cache
= 2 × layers × KV width × 4,096 tokens × 34/32 bytes (8-bit) and overhead = 0.40 GiB. Bytes per type come from
gguf-py (e.g. Q4_K: 144 bytes per 256 weights = 4.5 bits). A genome is valid if 0.99 × B ≤ RAM ≤ 1.005 × B GiB.

### Smarter rounding (`src/panther/rounding.py`, in the store step)

For every matrix W (rows × input columns): H = mean of x xᵀ over 65,536 calibration tokens (128 × 512), x being the
matrix's input in the original model (q/k/v share one H, gate/up share one). Damping: H += 1 % × mean(diag H) × I;
inputs that are always 0 get weight 0. U = upper Cholesky factor of H⁻¹. Then, 256 columns at a time (one
llama.cpp super-block): Q = llama.cpp's own quantizer applied to those columns (importance = diag H); the error
E = (W_chunk − Q) · U_cc⁻¹ is pushed onto the columns not yet quantized: W_rest −= E · U_c,rest. The codes overwrite
the type copy in place. Same file, size, speed.

### Offline error (`error` step)

error(m, t) = Σ over input columns c of imp_c × Σ over rows (w − ŵ)², with ŵ the matrix in type t (as rounded) and
imp_c the importance matrix (mean squared input of column c). Cheap; used only to repair genomes and as the
search's surrogate features, never as the score.

### The score (fitness), lower is better

score = flips % + 10 × drift

- **flips %**: on the GA's 900 selection questions (MMLU 300, ARC 200, HellaSwag 200, WinoGrande 200; llama.cpp picks
  the same questions every time, seed 1), the share of questions whose right/wrong differs from the original's,
  averaged over the 4 tasks. Example: 6 % → 6.
- **drift**: mean KL divergence per token between the genome's and the original's next-word probabilities
  (llama-perplexity --kl-divergence) on the original's own answers: its 200 GSM8K solutions (5-shot prompt, greedy,
  up to 256 tokens) interleaved with multiple-choice selection questions and their answers, 16 chunks of 2,048
  tokens. Example: 0.15 → 1.5.
- Example total: 7.5. Accuracy is recorded alongside, never optimised directly (it rewarded luck).

### The genetic search (`ga` step)

- **Genome**: a type for each of the model's matrices (7 per layer + word table + output layer; any of the 14 types,
  the word table excepting the 4 that need importance data).
- **Repair** (keeps every genome inside the window): too big → among all "one matrix one step down" moves, pick at
  random among the 8 that lose the least offline error per byte saved; repeat. Too small → the same upwards (moves
  that stay under the top), the 8 that gain the most per byte added.
- **Start population** (16): the preset's layout (each matrix its preset type, or the nearest), every uniform type,
  random mixes (a random type ± 0–2 steps per matrix), then mutants; all repaired, duplicates removed.
- **Mutation**: 50 %: 1–3 random matrices ± 1 step; 30 %: 1–3 matrices ± 2 or 3 steps; 20 %: a whole group (one
  matrix role in one quarter of the depth) ± 1 step. Then repair.
- **Crossover**: for each layer (word table and output count as one more), all of that layer's matrices from parent
  A or B (50/50). Then repair.
- **A child**: 30 % (or while the population is small): a mutant of one of the 4 best (elitism); otherwise two
  parents, each the better of 2 random population members (tournament), then crossover (50 %) or crossover + mutation
  (50 %). Genomes already scored are never scored again.
- **Solver proposals** (from 8 scored genomes on, whenever the queue is empty): fit score ≈ Σ_g w_g × x_g + c by least
  squares over every scored genome, x_g being the summed offline error of group g (role × depth quarter, ~30 groups);
  weights clipped to stay positive. Then CP-SAT minimises Σ w_g × error(m, t) over all genomes inside the window, 4
  answers that differ in ≥ 3 matrices; they and 2 mutants of them are queued.
- **Steady state**: up to min(16, 80 GiB / (Q4_K copy + 4 GiB)) genomes scored at once (1.5B: 16, 7B: 9, 14B: 6);
  each is built from the type copies, books GPU memory (file × 1.1 + 4 GiB), is scored, then deleted. The best 16
  are kept.
- **Stop**: after ≥ 30 genomes, once 15 in a row improve the best by less than 0.2; at most 70. The 3 lowest scores
  go to the final step.

### Distillation (`src/panther/distill.py`, in the final step)

- **Teacher**: the 16-bit original. **Student**: our file, loaded into PyTorch (transformers) with every weight as
  its quantized value.
- **Data**: the original's answers text + the calibration text, cut into sequences of 512 tokens (up to 256). The
  teacher's 64 most likely next words and their log-probabilities are stored for every position.
- **What is trained**: for every matrix, one number θ per block of 256 weights of a row, the effective weights
  being W × exp(θ) (θ starts at 0, so the student starts as the file); plus every norm weight. The whole numbers
  never change.
- **Loss**: at every position, KL(teacher ‖ student) over the teacher's top 64 (teacher probabilities renormalised
  over them; the student's from its full softmax). Adam, 300 steps of 4 sequences; learning rate 2·10⁻³ for θ,
  10⁻⁴ for the norms; gradient checkpointing.
- **Written back**: every K and IQ type stores one fp16 scale d per block (Q2_K, Q4_K, Q5_K also a minimum dmin) and
  a block's values are proportional to it, so d ← d × exp(θ) (and dmin) at the block's byte offset; Q8_0's 8
  sub-blocks per 256 weights all take that factor. Norm tensors are overwritten.
- **Not tuned**: IQ1_M blocks (no single scale), Mistral's attention q/k matrices (llama.cpp reorders their rows);
  InternLM cannot be loaded by transformers, so its finalists are scored as built (the page says so).
- Measured on the 1.5B's 2-bit preset (100 steps): drift 0.506 → 0.358, same top word as the original 64.5 % → 71.9 %.

### The final step and the verdict

1. Each of the top 3 distilled, then scored on all selection questions (4,502 multiple choice) with the same score;
   lowest wins.
2. **Guard**: winner vs the real preset file on those questions, paired bootstrap of the accuracy difference; if
   the whole 95 % interval is below 0, the winner is rejected and the contest ends without a result.
3. **Held-out test**, never seen before: MMLU 1,000, HellaSwag 2,000, ARC 1,172, WinoGrande 1,267 (multiple choice)
   and GSM8K (written answers, 5-shot, exact final number). Accuracy = mean of the 5 task accuracies. Also drift on
   the selection text, perplexity on web text, flips against the original.
4. **Paired bootstrap**: 10,000 resamples of the questions within each task, the same questions for both files;
   the 2.5 % and 97.5 % quantiles of (ours − preset) give the 95 % interval. **WIN** if all above 0, **LOSE** if all
   below, **TIE** otherwise.

### The competitor (`preset` step)

llama.cpp's presets (IQ1_S … Q8_0, its own mixes) built from the same original and importance matrix. Real RAM =
the file's tensors + KV cache + overhead. Of the two largest that fit (≤ 1.005 × B), the one with the higher
held-out accuracy competes (choosing on the held-out test favours the competitor, never us). None fits → the
smallest competes, with more RAM than we have.

### The Pi

llama-server with the 4,096-token context and 8-bit KV cache, 4 threads; one 256-token prompt and 64 generated
tokens. Reading speed = prompt tokens per second, writing speed = generated tokens per second; peak RAM = the
process's VmHWM; the throttle state before and after.
