# Panther

A Raspberry Pi 5 has 8 GB of RAM. For each RAM budget (1, 3, 5 and 7 GiB) and several open language models (Qwen2.5
1.5B, 7B, 14B; Mistral Small 22B and 24B), Panther builds a llama.cpp file that fits the budget and should lose less
of the original model's quality than llama.cpp's best ready-made preset that fits. Each model-and-budget pair is a
**contest**, written like `14b@5`.

## Results so far

Every result below was measured before the pipeline review of 20 Nov 2026, with the old method: the verdict was
the accuracy difference from the preset, and from 13 Nov the final step distilled on text that held the selection
questions (see "Found" below). None comes from the reviewed pipeline.

**Rounding, per type** (13 Nov 2026, Qwen2.5 7B, every matrix in one type; drift = KL divergence from the
original's next-word probabilities, lower is better). Error-compensated rounding is what `src/panther/rounding.py`
writes into every type copy.

| Type | Bits/weight | Drift, llama.cpp rounding | Drift, error-compensated | |
|---|---|---|---|---|
| Q8_0 / Q6_K / Q5_K | 8.5 / 6.6 / 5.5 | 0.0011 / 0.0021 / 0.0048 | | near-lossless |
| Q4_K | 4.5 | 0.0157 | 0.0145 | −8 % |
| IQ4_XS / IQ4_NL / Q4_0 | 4.25 / 4.5 / 4.5 | 0.020 / 0.020 / 0.030 | | worse than Q4_K |
| **Q3_K** | 3.44 | 0.134 | **0.064** | **−52 %**, equals IQ3_S |
| IQ3_S | 3.44 | 0.064 | 0.065 | no change |
| IQ3_XXS | 3.06 | 0.102 | | |
| Q2_K | 2.63 | 0.241 | 0.214 | −11 % |
| IQ2_XS | 2.31 | 0.333 | 0.302 | −9 % |
| IQ2_XXS / IQ1_M | 2.06 / 1.75 | 0.562 / 0.980 | | |
| MXFP4 / TQ2_0 | 4.25 / 2.06 | 0.235 / 13.2 | | do not work for normal models |

MXFP4 and TQ2_0 do not work for normal models, and IQ4_NL, Q4_0 and Q5_0 lose to the K types at the same size, so
the 14 types in `configs/search.yaml` leave them out.

**Distillation** (13 Nov 2026, the 1.5B's 2-bit preset): drift −29 %; the share of words where its most likely next
word is the original's, 64.5 % → 71.9 %.

**Contests:**
- InternLM2.5 20B (14 Nov 2026, search only, accuracy verdict): TIE at 5 GiB (+0.46 points) and at 7 GiB (+0.89).
  The model was then dropped for Mistral Small 22B: its model code does not run with the current transformers
  library, so it could get neither rounding nor distillation. Its logs and results are kept.
- Qwen2.5 32B (dropped 15 Nov 2026): loading its 16-bit weights (65.5 GB) for the rounding needs more RAM than a
  container gets (64 GB). Its contest, 32b@7, has no result (its competitor would have been IQ1_S: 54.43 % held-out
  accuracy, 7.70 GiB). The model and its files were removed.

**EfficientQAT** (a past experiment: Block-AP training of the weights block by block, then the block scales end to
end, on llama.cpp's own types; its code was removed on 20 Nov 2026 and stays in git history):
- Stage 1 on the 1.5B (15 Nov 2026, the IQ2_XS preset, 256 × 512 tokens, all blocks): web-text perplexity
  19.92 → 17.38 (original 10.72).
- The 14B @ 5 and 22B @ 7 QAT files were made before fix 79d9754 (20 Nov): until then a block's codes were picked
  again in fp32 when saved, while training and its checks picked them in bf16, so a block could be saved worse than
  it scored. Their results are true for the files as saved. The 14B @ 7 QAT run was stopped at block 17 on 20 Nov
  and removed (block 5: scored 0.01606, saved 0.01981; blocks 0-4 saved that way).

**On the Pi:**
- Weight repacking: on the Pi's ARM processor llama.cpp keeps a second, repacked copy of K-type weights unless
  repacking is off. The 7B @ 5 GiB winner peaked at 6.86 GiB with it, 4.69 GiB without; writing speed was the same,
  reading the prompt ~35 % slower. The Pi check runs with repacking off.
- Writing speed (16 Nov 2026, the 14B @ 5 QAT file): limited by the CPU decoding weights, not by memory. The Pi reads
  12.7–14.1 GB/s, enough for ~2.9 tokens/s, but every type family tops out near 20–22 billion weights a second (a
  14B: ~1.5 tokens/s), and four busy cores reach 80–85 °C within a minute and throttle (1.48 tokens/s cold, 1.25 over
  five answers). `ik_llama.cpp` (`fcf7b364`) reads prompts 6.6× faster than mainline (9.63 vs 1.47 tokens/s, no
  repacking) and writes 25 % faster with run-time repacking (`-rtr`, which halves reading). Speculative decoding with
  a Qwen2.5-0.5B draft (Q4_K_M, default settings) writes 1.79 tokens/s (+43 %, 68 % of guesses kept) but peaks at
  5.33 GiB: the draft adds ~0.6 GiB that batch size and draft context do not change. Within 5 GiB it fits only with a
  2,048-token context and a 4-bit KV cache (1.44 tokens/s, +15 %, answers changed). Longer forced guesses,
  auto-tuning, n-gram and suffix lookups gave nothing or lost. Any change of arithmetic path (`-rtr`, batching, a
  draft) changes some greedy answers by rounding; none was tested on the held-out questions.
- The 22B @ 7 QAT file (18 Nov 2026, `therachitkumar/panther-22b-7g-v2-qat`) with `ik_llama.cpp -rtr`: 0.96 tokens/s
  writing on the 64-token check and 0.81 over five answers (the contest winner on mainline: 0.67), 2.97 tokens/s
  reading (mainline 0.90), peak 6.60 GiB. Without `-rtr`: 5.12 tokens/s reading, 0.54 writing (`-rtr`: +78 % writing
  here, +25 % on the 14B).

## Running it

```bash
./panther run 14b@5          # a contest (model size @ RAM budget in GiB), detached: fetch -> run -> publish
./panther stop 14b@5         # stop that run
./panther pause | resume     # pause / resume every Panther container (the dashboard keeps running)
./panther status             # containers, active runs and their claims, GPU, disk, the last log lines
./panther build | test       # build the server image | the tests (here, then on the server in the GPU image)
./panther dashboard          # (re)start the read-only web dashboard
```

From your machine `./panther` syncs the code to the server and runs the same command there
([deploy/server](deploy/server/README.md) has the server's setup and `local.env`).

Locally it is a plain uv project: `uv sync` installs it, `uv run pytest` runs the tests. Dependencies change with
`uv add` / `uv remove` (or an edit to `pyproject.toml` followed by `uv lock`); `uv.lock` is committed, and the server
image installs exactly it (`uv sync --frozen`).

- **One run** is three containers in order: `fetch` (CPU, network), `run` (GPU, no network), `publish` (CPU,
  network). No container has both GPU and network.
- **The claim:** when it starts, before its three containers, the run claims the GPU memory and RAM it needs for its
  whole length (`data/locks/claims.json`). If that does not fit beside the active runs' claims (92 GiB GPU, 96 GiB
  RAM in all), it refuses to start and names the runs holding the rest. A claim ends with its run, however the run ends.
- **Failures are loud:** a failed step writes one `FAILED in <step>: <error>` line to `data/logs/panther.log` and the
  progress log, the traceback to its own log (`data/logs/<size>-b<budget>-<step>.log`), and stops the run.
- **Rerunning resumes:** every step returns at once if its output exists and was made from the same inputs, so
  `./panther run 14b@5` after a failure or a stop continues where it stopped.
- **GitLab first:** GitLab shares the server. A run waits to start while it is unhealthy, and a guard pauses every
  Panther container after 3 bad checks a minute apart and resumes after 5 good ones.
- **The dashboard** ([scripts/server](scripts/server/README.md)) shows runs, steps, verdicts, logs and the Pi's results.
- **The Pi** ([pi](pi/README.md)) checks each published winner: `nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`

## The pipeline

Terms: a **type** is one of llama.cpp's 14 storage formats (1.56 to 8.5 bits per weight); a **type copy** is the
whole model in one type; a **genome** gives every weight matrix its own type, and its file is assembled from the
type copies; the **preset** is the competitor, llama.cpp's best ready-made mix that fits. A file's **score** is
flips % (questions whose right/wrong differs from the original's) + 10 × KL (drift from the original's next-word
probabilities), lower is better; the **relative score** is ours ÷ the preset's × 100 (100 = the preset's damage).
Every number and formula: [src/panther](src/panther/README.md#in-depth).

The dataset has three splits, never mixed: **train** (built and trained on), **select** (the search and picking the
winner), **test** (the verdict only); benchmark questions are split into **selection** and **held-out** questions
([the dataset](#the-dataset-decision-6)).

| Container | Step | What | Data |
|---|---|---|---|
| fetch | fetch | the store's backup from Hugging Face, or else the original's weights at the pinned revision | |
| fetch | prompts | the chat prompts, once for all models | OpenAssistant, cut into train / select / test |
| run | claim | the run's GPU and RAM share (already held: the claim was made when the run started) | |
| run | check | the split check (decision 3); a failure stops the run | all splits |
| run | store | 16-bit GGUF, importance matrix, the 14 type copies with error-compensated rounding | train web |
| run | error | offline error of every matrix in every type (guides the search, never scores it) | |
| run | refs | the original's chat answers; its next-word probabilities on the select and test text; its answers to every question | select, test; all questions |
| run | preset | every preset that fits, scored; the most accurate competes and takes the held-out test | selection questions, select; held-out, test |
| run | ga | the genetic search; keeps the best 3 genomes | 900 selection questions, select |
| run | final | the best 3, each distilled and undistilled; the lowest score wins; the guard; the held-out verdict | distillation: train; winner and guard: all selection questions, select; verdict: held-out questions, test |
| run | bundle | the winner's file and manifest for the Pi | |
| publish | upload | the bundle to the public Hugging Face repo `<user>/panther-<size>-<budget>g-v2` | |
| publish | tidy | the store backed up to `<user>/panther-store-<model>`; its type copies removed unless another run of the model is active; the claim dropped | |

## Pipeline review (from 20 Nov 2026)

The pipeline was rethought stage by stage; each decision is recorded here.

**Found:** from 13 Nov (commit 4832b55) the final step distilled each finalist on `answers-<model>.txt`, which holds
the selection GSM8K questions with the original's answers and the selection multiple-choice questions with their
correct answers. The finalists were then ranked, and the guard run, on those same selection questions: a distilled
file was scored on questions it had trained on. The held-out test shares no question or text with any of it
(checked on the server's splits), so every verdict is a true measurement of the file that was tested; which file
was picked, and whether the guard was passed, may have been affected.

| # | Stage | Decision |
|---|---|---|
| 1 | Data | Each pool has one job. Calibration text is what things are built and trained on; selection questions are only scored on (search, picking the winner, guard); held-out questions and text are used once, for the verdict. Nothing trains on selection or held-out data. |
| 2 | Final step | Distillation trains on the original's own answers to general prompts (instructions, chat, everyday maths) from a pool in no other split, mixed with calibration text. The targets stay the original's probabilities; no dataset answers are trained on. Practice is for real use, not the benchmarks' format. |
| 3 | Data | The splits stay as they are (their builder was removed on 11 Nov, 87289e1). A short check, run at the start of every run, confirms the pools on the server share no question or text and match the hashes in their manifest. |
| 4 | Store | The importance matrix and our error-compensated rounding stay on the calibration web text alone (the preset uses the same importance matrix). |
| 5 | Error | The offline error estimate stays as it is: it reads only the type copies and the importance matrix, and only guides the search (repair, solver proposals); every genome is still measured. |
| 6 | Data | One dataset, split into train, select and test (below). KL and perplexity are measured only on its splits; flips and accuracy only on the benchmark questions. Nothing trains on anything but the train split. |
| 7 | Preset | The competitor is the most accurate on the selection questions of every llama.cpp preset that fits the budget, whatever its size (ties: lower KL on the select text). The held-out questions are no longer used to choose it, only for the verdict. If none fits, the smallest preset, as before. |
| 8 | Search | The score stays flips % + 10 × KL (flips on the 900 search questions, KL on the select text). It is reported relative to the preset, scored the same way: our score ÷ the preset's × 100 (100 = the preset's damage, 0 = none, above 100 = worse). The search itself is unchanged: dividing by one number does not change which genome is best. |
| 9 | Search | Early stopping as in ML training, with the minimum improvement set by the measurement's own noise: a genome counts as an improvement only if it beats the best by more than 1 standard error of the difference (paired over the same questions, from the questions where the two disagree; plus KL's uncertainty over its chunks). Stop after 15 genomes in a row without one (patience); at least 30, at most 70. The best 3 are re-scored on all selection questions in the final step, which corrects for the best of many noisy scores looking better than it is. Replaces the hand-set 0.2. |
| 10 | Search | **The fitness**, used by the search and to pick the winner: **score = flips % + 10 × KL** (as in 8), lower is better, reported relative to the preset. **Flips %** on benchmark selection questions (a flip needs a correct answer, which only benchmarks have): the search uses the same 900 (MMLU 300; ARC, HellaSwag, WinoGrande 200 each) with the first look (a third first; dropped if already ~2 standard errors worse than the 16th best), no GSM8K (measured in the held-out verdict). **KL** on our dataset's select split (chat and web). **Same top word %** (the share of words where the file's most likely next word is the original's), from the same KL run at no cost, is reported next to KL, in the search and the held-out test, and is not in the score. A "kept %" term (how far each answer stays identical) was considered and dropped: KL already measures it (a top word only changes where the probabilities move), it would add noise, and it needed an image rebuild or generation. |
| 11 | Final | The search's best 3 genomes are each built twice, distilled and undistilled (distillation helps most at tight budgets and can hurt near-lossless files, so the measurement decides). The 6 files are scored with the fitness (10) on all ~4,500 selection questions and the select split; the lowest score wins. Clean under 1 and 6: distillation trains only on the train split. |
| 12 | Final | The guard no longer stops a contest. The winner is compared with the preset on all selection questions by the verdict's rule (13); if ours is clearly worse (the relative score's 95 % interval entirely above 100) that is logged as a warning, and the held-out test runs anyway, so every contest ends in a reported WIN, TIE or LOSE. Before, a contest that would lose ended with no verdict, so the record showed only the contests that passed. |
| 13 | Final | **The verdict is the relative score on held-out data**: our fitness ÷ the preset's × 100, with flips on the held-out questions and KL on the test split (100 = the preset's damage, lower is better). Its 95 % interval comes from the same standard error the search uses (9): ±1.96 standard errors of the difference, from the per-question flips (paired) and the two KL runs. WIN if the whole interval is below 100, LOSE if it is all above, TIE otherwise. Accuracy, its difference from the preset, perplexity and same top word % are reported next to it and decide nothing. One rule throughout: the search, picking the winner, the guard and the verdict all use the same relative fitness (the guard on selection data, the verdict on held-out data). Replaces the accuracy difference with a bootstrap interval. At 95 %, about 1 contest in 20 where the two files are truly equal still comes out WIN or LOSE by chance. |
| 14 | Pi | The Pi's result line says FITS or OVER: whether the real peak RAM is within the bundle's budget. Reported on the dashboard next to the verdict; it does not change the verdict. |
| 15 | Final | Reported next to the verdict, deciding nothing: **accuracy kept**, each file's held-out accuracy ÷ the original's × 100, for ours and the preset; and whether the verdict stays the same with the KL weight at 5 and 20 instead of 10 (the weight is a choice, so its effect is shown). The relative score already places the original at 0 and the preset at 100. |
| 16 | Data | Every stage reads the whole of its split, never a part of it. The select split is made smaller instead (~100k tokens: the 150 chat prompts' answers and whole web documents up to the rest), so the search's KL reads all of it at today's cost. The held-out test reads the whole test split. Distillation covers the whole train split: every sequence, in a random order (fixed seed), for the same 600 updates as before (about 1.6 passes), so it takes about as long. Web documents left out of the smaller select split are used by nothing. |
| 17 | QAT | The QAT experiment's FineWeb-Edu sample (Stage 1: 4,096 × 2,048 tokens, its early-stopping checks 64 × 2,048; Stage 2: 4,096 × 4,096) counts as part of the train split: a web-only extension used only by QAT, about 60× larger than the rest of the train split. It shares no document with the select or test splits, and the split check confirms that. QAT's files are judged like a winner (guard on selection data, verdict on held-out data). The QAT code was removed afterwards (20 Nov). |

### The dataset (decision 6)

**The rule:** KL and perplexity are measured only on splits of this dataset. Flips and accuracy are measured only on
the benchmark questions. Nothing trains on anything but the train split.

Two kinds of text, each cut three ways with no overlap:

| | Train | Select | Test |
|---|---|---|---|
| **Chat**: OpenAssistant prompts (pinned, Apache-2.0) answered greedily by the original model in its own chat template | 300 prompts | 150 prompts | 150 prompts |
| **Web**: FineWeb-Edu documents (pinned) | 300k tokens | whole documents (fixed random order) until chat + web reach 100,352 tokens (49 × 2,048); the rest of the 150k unused | 150k tokens |

- **The web part** is the former calibration, selection and held-out web text, already disjoint by document
  (`calibration.txt`, `selection.txt`, `test_web.txt`).
- **The chat part:** prompts are split by a fixed seed. A prompt sharing 8 or more consecutive words with any
  benchmark question is dropped before splitting.
- **The answers are per model,** since each model is its own teacher.

**Which split is used at which stage:**

| Stage | Train | Select | Test |
|---|---|---|---|
| store: importance matrix, rounding | web part (decision 4) | | |
| refs: the original's probabilities recorded | | both parts | both parts |
| ga: the score's KL | | both parts | |
| final: distillation | both parts | | |
| final: picking the winner, the guard (the score's KL) | | both parts | |
| final: held-out KL and perplexity | | | both parts |

**Benchmark questions** (flips and accuracy only):

| Stage | Selection questions | Held-out questions |
|---|---|---|
| preset: choosing the competitor | ✓ | |
| ga: the score's flips | ✓ | |
| final: picking the winner, the guard | ✓ | |
| final: the verdict, accuracy reported | | ✓ |

The GSM8K few-shot examples stay as part of how a GSM8K question is asked.

**Where it is built and stored** (on the server under `data/splits/text/`):

1. `prompts` (in the fetch container, network): downloads OpenAssistant at the pinned revision, filters, splits,
   and writes `prompts.json` (each prompt's split, how many were dropped). Made once, shared by all models.
2. `refs` (GPU, no network): the original answers every prompt, written as `chat-<model>-<split>.txt`; the select
   text is `select100k-<model>.txt` (chat + web, cut to 100,352 tokens), the test text `test-<model>.txt` (chat +
   all of the test web text); the original's probabilities on them are `data/reference/<model>/select100k.kld` and
   `test.kld`. The select text's name carries its definition, so an older select text, reference or cached KL is
   never reused.
3. The check (decision 3), the `check` step at the start of every run: the three web files and both question splits
   match their manifests' hashes; no web document, question or chat prompt is in two splits; no prompt shares 8
   words with a benchmark question.

**Archived, not deleted:** `bench.txt`, `answers-<model>.txt`, the sensitivity and agreement text,
`gen_prompts.json`, WikiText, the MMLU few-shot examples.

## The code

| Module | What |
|---|---|
| `panther` | the one command (bash): sync, claim, containers, stop, pause, status, build, test, dashboard, the GitLab guard |
| [src/panther](src/panther/README.md) | the pipeline (`python -m panther.run`): data, store, measurement, search, final step, distillation, rounding |
| [scripts/server](scripts/server/README.md) | the read-only web dashboard |
| [pi](pi/README.md) | the Pi's check of each winner |
| [deploy/server](deploy/server/README.md) | the container image, the server's setup and its removal |
| [configs](configs/README.md) | models (pinned revisions), budgets, the RAM model's settings, types |
| `tests` | `./panther test`: parsing, the RAM model, the search's operators, the relative score, the claim and the FAILED path |

## Related work

- Choosing a type per layer by search: **EvoPress** (ICML 2025, evolutionary search with KL fitness); in GGUF,
  llama.cpp's `--target-bpw`, Unsloth Dynamic quants.
- Better rounding: **GPTQ**; **GSQ** (2026, learned rounding written back into GGUF Q2_K); **YAQA** (2025, rounding
  against the whole model's KL); **GuidedQuant** (ICML 2025). The strongest 2-bit methods (QTIP, PV-Tuning, AQLM) need
  their own formats.
- Training quantized models: **EfficientQAT** (block-wise training of the weights, then the quantization scales end
  to end); tried here as a past experiment (results above).
- Measuring: **Accuracy is Not All You Need** (NeurIPS 2024): flips and KL over accuracy.
- What Panther adds: allocation search over error-compensated type copies, K and IQ types, distillation into stock
  GGUF, RAM budgets that include the KV cache, checked on a real Pi, and a search score that does not reward luck.
