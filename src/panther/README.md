# src/panther

The pipeline. `./panther` runs it in containers as `python -m panther.run KEYWORD SIZE BUDGET`; the steps and the
data splits each uses are on one page in the [project README](../../README.md#the-pipeline).

| File | What | Main functions |
|---|---|---|
| `run.py` | The entry point: one keyword per container (`claim`, `fetch`, `run`, `publish`); the claim; the steps in order; any error becomes one FAILED line and exit code 3 | `claim`, `fetch`, `run`, `publish`, `tidy`, `main` |
| `data.py` | The dataset: the chat prompts, the split check, the select text | `prompts`, `check`, `select_text` |
| `store.py` | A model's store: the 16-bit original, its importance matrix, the 14 type copies; backed up to and restored from Hugging Face, type copies removed when no run of the model is active | `fetch`, `build`, `backup`, `clean` |
| `measure.py` | llama.cpp runs (KL, perplexity, multiple choice, WinoGrande; GSM8K and chat answers by llama-server), each output checked and cached by label; the original's reference outputs; a file's score and held-out test; the relative score, its standard error, the verdict | `run`, `refs`, `score`, `test`, `versus`, `beside` |
| `search.py` | The offline error, the preset (the competitor), the genetic search with CP-SAT proposals, building a genome's file from the type copies | `error`, `preset`, `ga`, `Genes`, `solve`, `build` |
| `final.py` | The final step (finalists distilled and not, the winner, the guard, the verdict), the Pi bundle | `final`, `bundle` |
| `blockap.py` | The optional QAT finalist's training (EfficientQAT's Block-AP): block by block, latent weights through the fake quantizers against the original's block outputs; codes written into a copy | `train` |
| `fakequant.py` | Fake quantizers for the 14 types: file bytes ↔ scales and codes, decoding bit for bit as llama.cpp, gradients straight through | `unpack`, `pack`, `encode`, `fake` |
| `distill.py` | Distillation of a finished file: block-scale multipliers and norms trained against the original, written back into a copy | `distill`, `teacher_tops` |
| `rounding.py` | Error-compensated rounding (GPTQ) with llama.cpp's own quantizer, written into the type copies in place | `blank`, `round_copies`, `gptq` |
| `model.py` | A model's matrices, bytes per type, the RAM model; what every step shares: paths, the progress log, hashes, locks, the run's GPU and RAM share (`hold`) | `Model`, `note`, `record`, `sha`, `lock`, `hold` |

**Where a run writes** (under `data/`): `store/<model>/` (the store); `rewrite/<model>/error.json`;
`reference/<model>/` (the original's probabilities); `results/<model>/` (one JSON record per llama.cpp run, its
output in `logs/`); `rewrite/budget-B/preset-<model>.json`, `list-<model>-v2.json` (the best 3),
`winner-<model>-v2.json` (the verdict and every held-out number); `rewrite/<model>/tuned/` (the winner's file);
`pi_bundle/budget-B-<size>-v2/`;
`logs/progress-b<B>.log` (the steps' notes) and `logs/panther.log`.

**Nothing is measured or built twice, and nothing is reused that was made otherwise.** A step returns at once if
its output records the same `inputs` digest. A step's inputs hold every constant of the modules it depends on
(`model.consts` reads them from the source, so none can be left out), the llama.cpp commit (`model.llama_commit`, from
`/opt/llama.cpp/.git/HEAD`), the RAM model's settings and the step's rule number (`model.RULES`, raised whenever its
written-out logic changes). A llama.cpp result is cached by label and its key (`how`): what was asked, the hash of
the task file, text or reference, the llama.cpp commit, its settings and measure's rule; a stored record is reused
only if its key is exactly the one asked for. A label carries what the file is: `g:<digest of the genome and the store's hashes>`,
`preset:<type>:<file hash>`, `d:<file hash>` (distilled), `original:<16-bit file hash>`. File hashes are kept beside
the file (`<file>.sha256`) while its size and time are unchanged. Files made once and reused (the chat answers, the
joined texts, the `.kld` references, GSM8K results, preset files) record what they were made from in
`<file>.inputs` (with the file's size and time) and are made anew when that differs. A genome's file is built in the
run's own `rewrite/<model>/tmp-b<B>/` and deleted after it is scored.

## In depth

### Types

Every type cuts a matrix row into blocks (256 weights; Q8_0: 32) and stores per block a 16-bit scale and each weight
as a small code; IQ types store groups of 8 or 4 weights as one pattern from a fixed grid. The 14 types, low to
high: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, IQ3_XXS, Q3_K, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0 (1.56 to
8.5 bits per weight). Any matrix may take any of them, except that IQ1_S, IQ1_M, IQ2_XXS and IQ2_XS need
importance-matrix data, which the word table never has and the output layer has only with
`imatrix_process_output` (`configs/models.yaml`). Where a type copy cannot use its type for those two, it holds Q8_0.

### The RAM model

**RAM = norms and biases (32-bit) + Σ every matrix's bytes in its type + KV cache + overhead**

- Bytes per type are exact (gguf-py's block sizes; Q4_K: 256 weights in 144 bytes = 4.5 bits).
- KV cache = 2 × layers × KV width × 4,096 tokens × 34/32 bytes (8-bit cache); overhead 0.40 GiB.
- A genome is valid only if **0.99 × B ≤ RAM ≤ 1.005 × B** GiB. That leaves about 3.0 bits per weight for the 1.5B
  at 1 GiB, 2.8 for the 7B at 3, 2.5 for the 14B at 5, 2.4 for the 22B and 2.3 for the 24B at 7.

### The claim

All claims together stay within 94 GiB of GPU and 100 GiB of RAM (`run.CAP_GPU`, `run.CAP_RAM`; the host has
~125 GiB, the rest stays for GitLab). A run claims 1 / `run.SIDE` of them for its whole length; `SIDE` is 1, so a run
has the whole machine and the next waits (two 14B runs could not share the GPU anyway). A model whose transformers
load (2 × parameters + 20 GiB of GPU, + 8 GiB of RAM: `model.trained`) does not fit a claim is refused. A claim
counts while its `data/locks/run-<id>.lock` is held. Inside the run every llama.cpp process, rounding pass and
distillation waits for its share of the claim less 20 GiB kept for the run's own processes (`run.RESERVE`,
`model.hold`): a file's llama.cpp runs take one booking, 1.1 × its file + 4 GiB of GPU and 5 × 4,096 × the
vocabulary × 4 bytes (multiple choice's buffer) + its file + 1 GiB of RAM, and run one after another; as many
genomes are scored at once as such bookings fit (at most 8: about 4 for a 14B). Rounding books the whole claim;
distillation books the transformers load, so files already built are scored beside it.
A run refused at its claim never started: a "refused" line, not a FAILED one.

### Error-compensated rounding (`rounding.py`)

For each layer matrix **W** (rows × input columns), in every type copy:

1. **H** = the mean of **x xᵀ** over 65,536 tokens (128 × 512) of the train web text, x being what the original
   feeds into that matrix. q/k/v share one H; gate/up share one.
2. Damping: + 1 % of H's mean diagonal; inputs that are always 0 get weight 0.
3. **U** = the upper Cholesky factor of H⁻¹.
4. 256 columns at a time: **Q** = llama.cpp's own quantizer (`ggml_quantize_chunk`, with the importance matrix) on
   those columns; error **E = (W_chunk − Q) · U_cc⁻¹**; then **W_rest −= E · U_c,rest**: the error is pushed onto the
   columns not yet quantized.
5. The codes overwrite the type copy in place: same type, size and speed.

The word table and output layer are quantized by llama.cpp without compensation (`blank`).

### Offline error (`search.error`)

**error(m, t)** = the expected squared change of matrix m's output per token when it takes type t, ŵ being m in
type t's copy:

- a layer matrix: **Σ over columns importance_c × Σ over rows (w − ŵ)²**, importance_c the mean squared input of
  column c (the store's importance matrix);
- the output layer: the same, its importance from a second importance file, `<model>.imatrix-output.gguf`
  (made beside the store's with `--process-output`; only this step reads it);
- the word table (its output is the row of the token read): **Σ over rows freq_r × Σ over columns (w − ŵ)²**,
  freq_r the share of token r in the train web text (the model's tokenizer); a tied word table is also the output
  layer and takes both terms (llama-imatrix records that use under `token_embd.weight`).

Missing importance data is an error. `error.json` is made anew when the store, the second file or the train text
changes. It only guides repair and the solver.

### The score

**score = flips % + 10 × KL**, lower is better.

- **Flips %**: the share of questions whose right/wrong differs from the original's, the mean over the tasks. In the
  search: 900 fixed selection questions (MMLU 300, ARC, HellaSwag, WinoGrande 200 each). In the final step and the
  preset step: all selection questions (n = 0, every question in the files).
- **KL**: llama.cpp's mean KL divergence per token from the original's recorded probabilities, on the select text
  (`select100k-<model>.txt`): the original's greedy answers (up to 256 tokens, in its chat template) to the 150
  select prompts, then whole select web documents in a fixed random order (seed 0) until 100,352 tokens (49 × 2,048).
  It is read in chunks of 2,048 tokens; a last partial chunk is not read (the count is logged). llama.cpp scores only
  the second half of each chunk (1,023 tokens, after 1,024 of context), so about half the tokens are scored (49
  chunks: 50,127 of 100,352). Each chunk's KL is recovered from the running mean printed after it (`kl_chunks`); a
  missing chunk line, or a last one that is not llama.cpp's mean, is an error.
- **Same top word %** comes from the same KL run and is reported, not scored. Accuracy is recorded, never optimized.
- **Relative score** = score ÷ the preset's score × 100, the preset scored on the same questions and text (the 900
  in the search, all selection questions in the final step): 100 = the preset's damage, 0 = none.
- **Standard error of a difference** (`measure.score_se`): per task, the standard error of the mean of the two files'
  paired flip differences (only questions where they disagree add to it), combined over the tasks as the SE of
  their mean; combined, as independent errors, with 10 × the SE of the mean of the two files' paired per-chunk KL
  differences (`measure.kl_se`: the same text and reference, so chunk k is the same text for both; about 49 chunks
  on the select text, about 110 on the test text).

### The preset (`search.preset`)

llama.cpp's mixes (IQ1_S … Q8_0, 22 of them) are made from the same 16-bit original and importance matrix; a mix
clearly too big by its nominal bits per weight is not made. Real RAM = the file's tensor bytes + KV cache + overhead.
Every mix with RAM ≤ 1.005 × B is scored on all selection questions and the select text; the most accurate there
competes (accuracy to 0.1 point; ties: lower KL). If none fits, the smallest made competes, with more RAM than we are
allowed (`over_budget` in its record). Only it takes the held-out test; preset files no budget chose are deleted once no other run of the model is active. Its score
on the search's 900 questions (`score_fit`) and on all selection questions are the bases of the relative score.

### The genetic search (`search.ga`)

- **Genome**: one type per matrix (7 per layer, the word table, the output layer if not tied), inside the RAM window.
- **Repair**: too big → among all "one matrix one step down" moves, rank by offline error added per byte saved, pick
  at random among the best 8, repeat until it fits; too small → the same upwards (error removed per byte added, never
  past the limit).
- **Start population**: the preset's layout (each matrix its preset type, or the nearest allowed), every uniform
  type, random mixes (a random type, each matrix shifted −2 to +2 steps) up to 10, then mutants of the first; all
  repaired, duplicates removed, up to 16.
- **Mutation**: 50 %: 1–3 random matrices ±1 step; 30 %: 1–3 matrices ±2 or ±3 steps; 20 %: a whole group (one matrix
  role in one quarter of the depth) ±1 step. Then repair.
- **Crossover**: per layer, all its matrices from parent A or B (50/50); the word table and output layer count as one
  more layer. Then repair.
- **A child**: 30 %: a mutant of one of the 4 best; otherwise two parents, each the better of 2 random population
  members, then crossover (50 %) or crossover + mutation (50 %). A genome already scored is never scored again.
- **Solver proposals** (from 8 scored genomes on, whenever the queue is empty): fit **score ≈ c + Σ_g (w + d_g) × x_g**
  over every scored genome, x_g = the summed offline error of group g (role × depth quarter, about 30 groups), each
  scaled by its mean; the d_g are shrunk towards 0 (ridge regression, its strength chosen by leave-one-out among
  10⁻³…10³), so with few genomes every group weighs about the same; a weight w + d_g under a twentieth of the mean is
  raised to it. CP-SAT then finds 4 genomes inside the window that minimize **Σ w_g × error(m, t)**, each at least 3
  matrices from the others; those 4 and mutants of the first 2 are scored next.
- **Steady state**: as many genomes at once as the claim holds (each counted as a file of the budget's size, at most
  16). The best 16 are the population.
- **First look**: once the population is full, a new genome is first scored on the first third of the questions
  (MMLU 100, the others 66: llama.cpp draws a task's questions in a fixed order, so these are the first third of the
  900) and the full KL. It is compared with the population's worst on the same questions and chunks, paired, and
  dropped if worse by more than 2 standard errors of the difference.
- **Length**: 70 genomes (dropped ones count), with no early stopping: the final step's re-scoring of the best 3 on
  all selection questions decides among noisy scores. Each genome's origin (start, mutation, crossover,
  crossover+mutation, solver, solver mutant), score and whether it was dropped are in the list record (`genomes`),
  the finalists' origins beside them. The 3 lowest go to the final step.

### Distillation (`distill.py`)

- **Teacher**: the 16-bit original. **Student**: our file, loaded into PyTorch with every weight at its quantized
  value.
- **Data**: the train split only: the original's chat answers to the 300 train prompts, then all of the train web
  text, in sequences of 512 tokens. For every position the teacher's 64 most likely next words and their
  probabilities are stored, once per model and train text (`reference/<model>/teacher-top64.pt`), for every
  distillation of that model.
- **Trained**: for every layer matrix, one number **θ per 256 weights of a row**, effective weights **W × exp(θ)**
  (θ starts at 0; computed in fp32); plus every norm weight. The codes never change; the word table and output layer
  are not tuned.
- **Loss**: KL(teacher ‖ student) over the teacher's top 64 (renormalized), at every position. Adam, 600 steps of 2
  sequences, every sequence used once in a random order (seed 0) before any repeats (about 1.6 passes); learning rate
  0.002 for θ, 0.0001 for norms; activations recomputed instead of stored.
- **Written back**: every K and IQ type stores one 16-bit scale d per block, and a block's values are proportional to
  it, so **d ← d × exp(θ)** at the block's byte offset (Q2_K, Q4_K, Q5_K also scale dmin; Q8_0's 8 blocks per 256
  weights all take the factor). IQ1_M has no single scale and is left as it is. Norms are overwritten.
- **Llama-style models** (Mistral): llama.cpp reorders the rows of attention q/k when converting, so their multipliers
  are put into the file's row order; a check first confirms the reordered weights match the file row for row (if
  not, those matrices are not tuned).

### The final step and the verdict (`final.py`)

1. Each of the best 3 genomes is built undistilled (the same file the search scored, so its results are reused) and
   distilled (one distillation at a time, while the files already built are scored beside it). With `qat[=N]`
   (decision 22) a 7th: the best genome's undistilled file trained by Block-AP on the N QAT samples
   (`rewrite/<model>/qat/tokens-N.npz`, drawn in the fetch container), then distilled. The files are scored on all
   selection questions and the select text; the lowest score wins.
2. **Guard**: the winner's relative score against the preset on all selection questions and the select text, with
   its 95 % interval; if the whole interval is above 100, a warning is logged and stored (`guard.warning`). The
   held-out test runs anyway.
3. **Held-out test**, for ours, the preset and the original side by side: MMLU, HellaSwag, ARC and WinoGrande (every
   held-out question), GSM8K (5 worked examples, greedy, 256 tokens; the number after `####` must match), KL and same
   top word % on the test text (`test-<model>.txt`: the original's answers to the 150 test prompts, then all of the
   test web text), perplexity on the same text.
4. **The verdict**: score = flips % (the 5 held-out tasks) + 10 × KL (test text), for ours and the preset; relative
   score = 100 × ours ÷ preset's; 95 % interval = 100 × (ours ± 1.96 SE) ÷ preset's, with the SE of the difference
   above. **WIN** if the whole interval is below 100, **LOSE** if all of it is above, **TIE** otherwise.
5. **Reported, deciding nothing**: accuracy (the mean of the 5 task accuracies) and ours − preset in points,
   perplexity, same top word %, accuracy kept (each file's held-out accuracy as % of the original's), and the
   relative score, interval and verdict again with the KL weight at 5 and 20.

The winner record (`winner-<model>-v2.json`) stores `relative`, `relative_ci`, `diff` and `ci` (the same minus 100),
`verdict`, `accuracy_diff`, `accuracy_kept`, `weight_check`, `guard`, `selection`, and `ours`, `preset_test`,
`original` (accuracy, KL, same top word, perplexity, flips, per task). Only the winner's file is kept.

**The bundle** (`final.bundle`): the winner's file and a manifest (hash, budget, predicted RAM, context, KV cache
type), uploaded by `publish` to the public repo `<user>/panther-<size>-<B>g-v2`. The Pi's check: [pi](../../pi/README.md).

## Tests

`./panther test` runs `tests/` on your machine, then in the GPU image on the server: llama.cpp output parsing, the
RAM model, the search's operators, the relative score, the claim and the FAILED path.
