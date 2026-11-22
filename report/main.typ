#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Panther: Genetic Algorithm-Driven Multi-Bit Quantization of LLMs for Efficient Inference on Resource-Constrained Hardware],
  abstract: [
    Deploying large language models on memory-constrained devices requires aggressive quantization, yet uniform low-bit quantization imposes a quality penalty that grows sharply as precision drops. This work pushes that trade-off: for a given model and a strict memory budget, it looks for the quantized representation that loses the least quality, and checks whether that beats the ready-made quantization recipes that fit the same budget.

    We formulate the problem as a search over mixed-precision (multi-bit) configurations, in which a quantization type is chosen independently for each weight matrix of the model rather than a single type applied globally. Each type is realized through error-compensated quantization (GPTQ @gptq), which rounds columns of a matrix while redistributing rounding error under a Hessian estimated from real calibration activations, so that the compensated low-bit representation better preserves the matrix's output. Candidates are then assembled by combining the per-type matrices without re-quantizing, making the search fast enough to be practical. The search itself is driven by a genetic algorithm: an initial population of per-matrix assignments is evolved through crossover and mutation, with each candidate's quality scored by a single fitness measure that couples benchmark answer degradation with a divergence term over next-token distributions. A lightweight memory model — file size plus runtime activation buffers — enforces the budget, and the evolved candidate is judged against the best fitting preset.

    The hardware and model families used here are a testbed; the method is meant to carry over to other models and devices.
  ],
  authors: (
    (
      name: "Jia Khot",
      department: [Computer Science and Engineering],
      organization: [2310110491],
      email: "jk637@snu.edu.in",
    ),
    (
      name: "Lalit Maurya",
      department: [Computer Science and Engineering],
      organization: [2310110164],
      email: "lm742@snu.edu.in",
    ),
    (
      name: "Rachit Kumar",
      department: [Computer Science and Engineering],
      organization: [2310110234],
      email: "rk551@snu.edu.in",
    ),
  ),
  index-terms: (
    "Multi-bit quantization",
    "Large Language Models",
    "Genetic Algorithms",
    "Resource-constrained hardware",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Large language models (LLMs) are now central to many natural-language systems, but their parameter counts—often in the billions—make them expensive to store and serve, and hard to run outside a well-provisioned data center. For many practical settings, and for inference on resource-constrained devices in particular, the binding constraint is not raw computation but _memory_: the model's weights, together with their runtime activations, must fit in a few gigabytes of RAM that are shared with the operating system and other processes. When a model does not fit, a system must either offload parts of it to slower storage @llminaflash or shrink it. We study the latter.

Quantization—representing each weight in fewer bits—is a standard way to shrink a model, and post-training quantization (PTQ), which needs no retraining, has become a common choice for LLMs @quantization_survey. A body of work has since made low-bit LLM quantization practical: near-lossless 8-bit inference @llmint8, activation-aware and outlier-smoothing methods @awq @smoothquant, error-compensated rounding that reduces each matrix's output distortion @gptq, and sparse-quantized representations @spqr. Nearly all of these methods, however, apply a single quantization scheme uniformly across the network.

The observation that motivates this work is that a network's layers are not equally sensitive to quantization. Some matrices tolerate aggressive low precision with little harm, while a few—the output head, the embeddings, and certain attention projections—dominate the quality loss. This asymmetry is well documented, and both second-order and search-based methods have exploited it: HAWQ derives per-layer precisions from the Hessian spectrum @hawq, while HAQ learns a hardware-aware bit-width policy through reinforcement learning @haq. Mixed-precision—or _multi-bit_—quantization, which assigns a different bit-width to each matrix, therefore offers more room to trade size against quality than a uniform scheme.

We adopt this mixed-precision view and make the deployment problem concrete: _given a model and a strict memory budget, find the assignment of a quantization type to each weight matrix that fits the budget and minimizes quality loss._ This is a large, discrete, non-convex search whose space is exponential in the number of matrices, but it is the kind of problem evolutionary search handles well. We search it with a genetic algorithm @goldberg_ga, evolving a population of per-matrix assignments through selection, crossover and mutation. Each candidate is scored by a single fitness measure that combines benchmark-answer degradation relative to the full-precision reference with a Kullback–Leibler divergence term over next-token probabilities; a lightweight memory model enforces the budget and rejects infeasible candidates.

Two choices make the search tractable and grounded in real hardware. First, every matrix is drawn from the fixed set of quantization types implemented by the `llama.cpp` inference engine @llamacpp, so a type means exactly what it will mean on the device. Second, rather than re-quantizing the model for each candidate, the model is pre-built once in every candidate type using GPTQ's error-compensated rounding @gptq; a candidate is then assembled by copying the chosen per-matrix bytes, turning each evaluation from a full quantization pass into a fast byte copy. The genetic algorithm can therefore explore a far larger region of the assignment space within a practical time budget.

The specific hardware and model families used in this study are a controlled testbed; the method is intended to generalize to any memory-constrained deployment target. Our contributions are fourfold: (i) a formulation of budget-constrained LLM deployment as a mixed-precision, per-matrix assignment search; (ii) an evaluation scheme that couples GPTQ-compensated per-type copies with byte-level candidate assembly, making evolutionary search tractable; (iii) a unified fidelity metric—benchmark answer flips plus a next-token divergence term—that scores searched candidates and ready-made recipes on a common scale; and (iv) an honest evaluation protocol that separates the search, winner and verdict data and judges the winning file against the best fitting preset with a paired 95 % interval. The remainder of the paper describes the quantization machinery and the search in detail, evaluates the method against baseline recipes, and discusses limitations and future work.


= Related Work

We review the prior work in two strands: post-training quantization for large language models, and multi-bit and mixed-precision quantization. We close by identifying the gap that motivates this work.

== Post-Training Quantization for Large Language Models

Quantization compresses a model by representing its weights, and sometimes its activations, in fewer bits, and it is a common remedy for the memory and compute cost of serving LLMs. The design space is broad: quantization can be applied after training (post-training quantization, PTQ) or folded into training (quantization-aware training, QAT); it can target weights only or weights and activations together; and its bit-width can be uniform across the network or assigned per-layer or per-matrix (mixed precision) @quantization_survey. Because retraining a large model is expensive and calibration data is often scarce, PTQ is now the usual approach for LLMs, and a succession of methods quantize LLMs to 3–4 bits with only modest degradation @gptq @awq.

_GPTQ_ is a one-shot, weight-only PTQ method built on approximate second-order information @gptq. It treats the layer-wise reconstruction $norm(W X - hat(W) X)_F^2$ as a least-squares problem in the weights, approximates its Hessian by the input second moment $X X^T$, and quantizes the columns of $W$ in order while propagating each rounding error onto the columns not yet quantized, in blocks of 128–256 columns with a Cholesky factorisation of the inverse Hessian. _AWQ_ takes a complementary, activation-aware view @awq: a small fraction of weight channels (roughly 0.1%–1%) carry disproportionately large activations and dominate the layer's sensitivity, so AWQ rescales those channels in a way that leaves the output unchanged but moves the important weights into the well-resolved part of the quantizer's range, found by a fast search rather than backpropagation. This idea — that a weight's importance is tied to the activation it sees — recurs in the importance-matrix machinery we use. Related methods smooth the activation outliers that make low-bit quantization hard @smoothquant, and sparse-quantized formats push 3–4-bit quality further @spqr @llmint8. What all of these share is a focus on one high-quality quantization at a fixed precision, rather than on choosing precisions across the model.

== Multi-Bit and Mixed-Precision Quantization

Layers are not equally sensitive to quantization, and assigning precision non-uniformly can beat a uniform scheme at a fixed budget. HAWQ formalizes this with second-order information: it estimates each layer's sensitivity from the spectrum of its Hessian and uses that ranking to allocate precisions, and it orders fine-tuning by the same signal @hawq. HAQ, in contrast, treats the allocation as a learning problem and uses reinforcement learning with direct hardware feedback to produce a bit-width policy specialized to a target accelerator @haq. Both demonstrate the value of mixed precision but operate in settings—image classification, convolutional architectures—far from decoder-only LLMs.

The closest precursor is _PTMQ_, which brings multi-bit support into the post-training setting @ptmq: rather than producing one model per precision, it calibrates a single model whose bit-width can be switched at runtime, in both uniform and mixed-precision modes, using a block-wise reconstruction with a Multi-bit Feature Mixer and a Group-wise Distillation Loss, and fusing normalisation layers so their statistics need not be recomputed on switching. For its mixed-precision allocation, PTMQ ranks each block's sensitivity with a KL-divergence metric and selects bit-widths along a Pareto frontier using a genetic algorithm under a model-size constraint. Two details are especially relevant to us: PTMQ adopts a genetic algorithm for exactly our budget-constrained allocation problem, and it measures sensitivity with KL divergence, the same quantity our fitness is built on. Its evaluation, however, is confined to convolutional networks and vision transformers; carrying the multi-bit idea to decoder-only LLMs, combined with LLM-specific quantization, is the gap this project investigates.

= Methodology

Panther is a linear pipeline of eight stages. This section describes each stage as it is implemented today, in pipeline order. Every stage is a tested Python module that shells out to existing tools—`llama.cpp`, Hugging Face Transformers, `gguf-py`, and OR-Tools—rather than reimplementing them, and every stage is idempotent: it skips any step whose output already exists, so a stopped run resumes where it left off. Failures are raised loudly, never caught and ignored.

== Problem Formulation

A _contest_ is a pair of a model and a RAM budget in GiB. The pipeline's goal is to produce a single `llama.cpp` file that fits the budget and loses less of the original model's quality than the best ready-made preset that also fits. A _genome_ $g = (g_1, dots, g_M)$ assigns to each of the model's $M$ matrices—the word table, the seven matrices of every layer, and the output layer—one of 13 quantization types, $g_i in T$, so a 7B model has $M = 198$ genes and the 24B has 282. A genome's memory cost is modelled as

$ "RAM"(g) = "file"(g) + "KV cache" + "overhead" , $

and a genome is _feasible_ only inside the window $0.99 B <= "RAM"(g) <= 1.005 B$, where $B$ is the budget; the lower bound forces the search to spend the budget it is given rather than leave RAM unused. The fitness of a genome is its score, defined below, and the task is to find the feasible genome of lowest score.

== Data (Part 0)

Every text and question the pipeline reads is built by code from pinned Hugging Face sources, in four sets plus QAT's text, so that no set shares a web document, a chat prompt or a question. Web text comes from FineWeb-Edu, each document assigned to one of five shards by hashing its id, so dropping one never moves another between sets. A document is dropped if it holds twelve consecutive words of any benchmark question, which keeps the answers out of the calibration and evaluation text; stems under eight words are not searched, because a short common phrase would otherwise erase every document that contains it. Chat prompts come from OpenAssistant, deduplicated and checked for overlap, and each model answers them itself in its own chat template, so the text matches the model being measured. Every question file is written in llama.cpp's own formats and read back before use to confirm it round-trips. @fig:datasets shows how the sets feed the stages.

#table(
  columns: 3,
  [Set], [Used for], [Questions],
  [GA], [the search's fitness], [MMLU 300, HellaSwag 200, ARC 200, WinoGrande 200],
  [winner], [choosing the preset and the winner], [MMLU 1,000, HellaSwag 800, ARC 800, WinoGrande 800],
  [distill], [importance matrix, GPTQ, distillation], [none],
  [verdict], [the verdict, once per contest], [MMLU 2,000, HellaSwag 2,000, ARC all, WinoGrande all, GSM8K 500],
)

#figure(
  image("diagrams/datasets.png", width: 100%),
  caption: [The four data sets and the stage each feeds. Nothing is trained on the search's, winner's or verdict data, and nothing is chosen with the verdict set.],
) <fig:datasets>

== The Model and Its Type Copies (Part 1)

This stage turns a model into the materials the search will assemble: a 16-bit original, an importance matrix, and one whole-model copy in each of the 13 types. The weights are downloaded once at a pinned revision (kept, so nothing is ever downloaded twice; Mistral's duplicate `consolidated*` files are skipped) and converted to a 16-bit bf16 GGUF, which from then on is "the original". `llama-imatrix` runs the original over the distill set's web text and records, for every matrix including the output layer (`--process-output`), the mean squared activation of each input column; this _importance matrix_ lets the data-dependent types spend more bits on the columns that matter,

$ I_j = 1/N sum_(n=1)^N x_(n j)^2 . $

Then, for each of the 13 types, `llama-quantize --pure` writes the whole model in that type, and GPTQ overwrites every layer matrix and the output layer with error-compensated codes. The 13 types are `IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, IQ3_XXS, Q3_K, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0`, with `IQ3_S` not included because it matches `Q3_K` in size. They are block-structured: each splits a weight row into blocks of 256 weights (32 for `Q8_0`) and stores one 16-bit scale $d_j$ plus a small integer code $c_(i j)$ per weight, so a weight is reconstructed as

$ w_(i j) approx d_j c_(i j) . $

The word table is a lookup with no input, so it has no importance data and the four lowest types would not quantize it; in those four copies it is stored as `Q8_0`, and the search never assigns it those types. The output layer is treated like any layer matrix—its error goes straight into the next-word probabilities—so it receives importance data and GPTQ alike.

GPTQ here is the mechanism of @gptq, with one twist: our implementation calls `llama.cpp`'s own quantizer (`ggml_quantize_chunk`) as its rounding primitive, so the codes written into a copy are byte-for-byte what the engine would produce, with no re-encoding step. The Hessians are collected by running the model under Hugging Face Transformers and hooking each matrix's input @transformers. GPTQ also records each matrix's output error in every type to an `errors.json`, which the search's repair uses as a per-byte cost. Throughout, the original stays in bf16 as published and the arithmetic runs in 32 bits, because llama.cpp's quantizer takes float32 and a 16-bit Cholesky factorisation would be too imprecise. @fig:assembly shows how the search later assembles a candidate from these copies.

#figure(
  image("diagrams/assembly.png", height: 3.5in),
  caption: [A candidate is assembled by copying each matrix's bytes from the pre-built, GPTQ-rounded type copy of its assigned type, so its size is known exactly and no re-quantizing happens during the search.],
) <fig:assembly>

== The Competitor: Presets and the RAM Model (Part 2)

The competitor for a contest is llama.cpp's best ready-made recipe that fits the budget. The 22 candidate presets are sized exactly with `llama-quantize --dry-run`—instant, and nothing is quantized—before any is built; those whose dry-run size fits are built, and their built file size is checked against the budget, since the device loads the whole file. Each fitting preset is scored by the fitness on the winner set, and the lowest score competes: the choice is made by the very measure we compete on, not by nominal accuracy. The RAM model is file size plus the KV cache plus a fixed 0.40 GiB of runtime overhead, where the KV cache holds keys and values for every layer at a 4,096-token context in 8 bits, its shape read from the GGUF's own metadata. A file fits if its RAM is at most 1.005 × the budget. If no preset fits, the smallest competes anyway and the result is tagged _unfair_, so very small budgets can still be tried.

== Fitness (Part 3)

One fitness is used everywhere—choosing the preset, ranking search candidates, and picking the winner—so every number is comparable. The divergence term is the mean Kullback–Leibler divergence of a file's next-token distribution $P$ from the original's $Q$,

$ D_"KL"(P, Q) = sum_(v in V) P(v) log (P(v))/(Q(v)) , $

and the flips term is the share of benchmark questions whose right/wrong answer differs from the original's, averaged per task. The two are combined with a fixed weight $lambda = 10$ into a single score

$ "score" = "flips%" + lambda dot D_"KL" , $

so answer flips dominate while the divergence breaks near-ties, and a candidate is reported against its preset as

$ "relative" = 100 dot ("score"_"ours")/("score"_"preset") , $

where 100 is the preset's damage and below 100 is less. Every candidate gets the full measurement—there is no cheap "first look" that drops unpromising files early—and nothing is cached: `score` measures and returns, while only the original's reference answers are recorded once, named by the content of the data they came from so new data never reuses old records. The KL is read over the second half of each 2,048-token chunk, llama-perplexity's own scoring convention. To keep the search fast, a file's five measurements run at once and several files are scored in parallel, under a shared GPU/RAM budget that reads the container's memory limit rather than the whole server's.

== Search (Part 4)

The search is a genetic algorithm over genomes, kept reproducible and inside the RAM window. The starting population repairs and scores the 22 preset layouts and the 13 one-type layouts, plus four layouts found exactly by OR-Tools CP-SAT minimising the summed output error from `errors.json`. Each round produces seven children and one solver proposal, scored eight at a time: roughly 30 % of children mutate one of the four best (moving one to three matrices up or down by one or two types), and the remaining 70 % are a two-point _slicing_ crossover that swaps a contiguous run of whole layers between two parents, keeping neighbouring layers—which tend to want similar precision—together. Every offspring is _repaired_ into the window by stepping the matrix whose type change costs the least output error per byte, and the solver's proposal is the genome of lowest summed error once each role's error is re-weighted by a ridge fit (pulled toward one) over the scores seen so far, so the solver and the genetic operators work together. The whole search is seeded, so the same seed reproduces the same result, and it stops after 15 candidates in a row with no new best, between 30 and 70 candidates in all. Each candidate's file is assembled in memory and deleted after scoring, and every score is recorded, so a stopped search continues exactly where it left off. @fig:search summarises one round.

#figure(
  image("diagrams/search.png", height: 4.8in),
  caption: [One round of the search: children are produced by mutation and two-point slicing, repaired into the RAM window, joined by a CP-SAT proposal, scored, and folded back into the population.],
) <fig:search>

== Distillation (Part 5)

Distillation is a lightweight training step that nudges each block of weights so a finalist's next-word predictions match the original's again, without changing its size or speed. It attaches one learnable multiplier to each block of 256 weights—a scale $exp(theta)$ with $theta$ initialised to zero—together with a small correction to every norm, and trains them under a KL loss over the teacher's 64 most likely next words plus a catch-all outcome,

$ L = D_"KL"( P_"teacher", P_"student" ) . $

The loss is the exact KL over those 65 outcomes. Because multiplying a block's weights by $exp(theta)$ is the same as multiplying its stored scale $d_j$ by $exp(theta)$, writing the result back is simply

$ d_j' = d_j dot exp(theta_j) , $

plus the norm shifts; the codes are never touched, so the file keeps its size and speed byte for byte. The output layer is trained but the word table is not, and Mistral's reordered q/k rows are handled both ways. Only the written file is ever scored, never the in-memory model. Measured on the test model, distillation cuts KL by roughly 70 % at low bit-widths (e.g. `Q2_K` 0.0288 → 0.0079).

== QAT (Part 6, optional)

Quantization-aware training goes beyond distillation: instead of only tuning scales, it retrains the weights themselves while keeping them on the type's grid. This stage implements EfficientQAT's first stage, Block-AP @efficientqat, which trains one transformer block at a time so the quantized block reproduces the original's output. Each matrix keeps a 32-bit latent copy of its weights that is rounded onto the file's grid in the forward pass (with the gradient passed straight through the rounding), and the codes nearest the latent weights are re-picked periodically; the scales stay as the file has them, since distillation tunes them afterwards. Four guarantees make the training honest: a single 32-bit code picker everywhere, an exact return to the best checkpoint, scoring from the saved bytes, and never saving a block worse than the file's. The stage is optional because its on-disk activation caches are large—at the paper's 4,096 samples, 172 GB for the 14B—so about 1,024 samples are used in practice, and it is off by default. On the test model, Block-AP alone cut `Q2_K`'s KL from 0.0294 to 0.0165 before distillation.

== The Winner and the Verdict (Part 7)

The winner is chosen on the winner set, never on the GA set where the finalists became finalists—partly by luck—and the verdict is taken once on the verdict set. The three search finalists are each built as searched and distilled (plus QAT's file if requested), all scored at once, and the lowest score wins; only its file is kept. The verdict then measures the winner and the preset at the same time, on the same questions and text, and reports our damage relative to the preset's. Because both files answer the same questions and read the same text, the comparison is paired, and the uncertainty of the relative score is the standard error of the paired difference,

$ "SE" = sqrt("SE"_"flips"^2 + (lambda dot "SE"_"KL")^2) , $

which grows only from questions where the two files disagree and from chunks where their divergences differ. The 95 % interval is the relative score plus or minus 1.96 standard errors, and the verdict is WIN, TIE or LOSE according to whether that interval lies wholly below, contains, or lies wholly above 100. A step-by-step reasoning benchmark (GSM8K, run through `llama-server`) rounds out the judgement, since reasoning is where heavy quantization most often fails first.

== Running a Contest (Part 8)

One Python entrypoint calls the stages in pipeline order, and one command runs a whole contest on the server. `python -m panther fetch` builds the dataset and downloads the weights in a container with network but no GPU; `python -m panther run` builds the model, chooses the preset, runs the search, and judges the result in a container with the GPU and no network, so no container has both. The image pins llama.cpp to the repository's commit and torch to a matching CUDA build, and every step skips what it has already made, so a run continues where it stopped. A per-contest log records progress, and the verdict line at the end is the single-number result.

== Pipeline summary

Taken together, the eight stages form a single, reproducible pipeline: given a model and a budget, it builds the materials once, searches the space of per-matrix type assignments under a memory budget, refines the best candidates with distillation and, optionally, quantization-aware training, and finally judges the winner against the best ready-made preset with a paired, interval-qualified verdict. Three design choices carry the whole. The type copies turn candidate evaluation into a byte copy, so the search can explore the exponential assignment space cheaply. A single fitness measure ranks every file—candidate, preset and winner alike—on one scale. And the disjoint data sets keep training, selection and judgement honest, so the final comparison is against the strongest preset that fits, on questions neither file has seen. The whole is summarised in @fig:pipeline.

#figure(
  image("diagrams/pipeline.png", height: 5in),
  caption: [The Panther pipeline, stage by stage, from the raw data to the verdict.],
) <fig:pipeline>

= Experimental Setup

Three instruction-tuned models were quantised and evaluated end to end, in three contests. Everything that touches a model — the download, the importance matrix, GPTQ, the genetic search, distillation and the optional QAT — ran on one GPU server; the resulting `llama.cpp` files were then loaded and timed on a Raspberry Pi 5.

== Models

#figure(
  table(
    columns: 4,
    [Model], [Parameters], [RAM budget], [Competitor preset],
    [Qwen2.5-1.5B-Instruct], [1.5B], [1 GiB], [`IQ2_S`],
    [Qwen2.5-7B-Instruct], [7.6B], [5 GiB], [`Q4_K_M`],
    [Qwen2.5-14B-Instruct], [14.8B], [5 GiB], [`IQ2_XXS`],
  ),
  kind: table,
  caption: [The three models and the contest each was run in. The competitor is the fitting preset with the lowest score on the winner set.],
) <tbl:models>

== The build server

#figure(
  table(
    columns: 2,
    [Component], [Specification],
    [System], [ASUSTeK ESC4000A-E12],
    [OS], [RHEL 10.2, kernel 6.12],
    [CPU], [AMD EPYC 9224 — 24 cores / 48 threads, 1.5–3.7 GHz, AVX-512],
    [Memory], [124 GiB + 31 GiB swap],
    [GPU], [NVIDIA RTX PRO 6000 Blackwell, 96 GiB (sm_120, 600 W, PCIe 5 × 16, driver 610.43, CUDA 13.3)],
    [Disk], [`/s3`: 931 GB XFS on a 3.6 TB SSD behind a MegaRAID 9540-8i],
    [Network], [1 Gb/s],
  ),
  kind: table,
  caption: [The server on which all calibration, search and training ran.],
) <tbl:server>

== The target device

The quantised files run on a Raspberry Pi 5 with 8 GiB of RAM and a 64 GB SD card. A contest's RAM budget is the Pi memory the file must fit into, and the Pi is where the decode and prefill speeds are measured: four threads, the 8-bit KV cache at a 4,096-token context, a 256-token prompt and 64 generated tokens, with peak RAM read from the server process. Each file is timed with mainline `llama.cpp` and with `ik_llama.cpp`, a fork with faster CPU kernels for the low-bit types, with and without its run-time repacking (`-rtr`).

== Software

Model work uses `llama.cpp` at the pinned commit `cd959bae` (tag v0.9.0): `llama-quantize` for the type copies and presets, `llama-imatrix` for the importance matrix, `llama-perplexity` for every fitness measurement, and `llama-server` for the chat answers and GSM8K. Hugging Face Transformers loads the models for the GPTQ Hessians and distillation, and OR-Tools CP-SAT proposes the search's candidates. Work runs in rootless Podman (5.8.2) with the GPU passed through CDI, split between a `fetch` container (network, no GPU) and a `run` container (GPU, no network).

= Results

Three contests were run end to end and judged once on the held-out verdict set: Qwen2.5-1.5B-Instruct at 1 GiB against `IQ2_S`, Qwen2.5-7B-Instruct at 5 GiB against `Q4_K_M`, and Qwen2.5-14B-Instruct at 5 GiB against `IQ2_XXS`. All three presets fit their budgets, so every contest is fair. The verdict — the winner measured against the competitor on questions and text neither file has seen — is a WIN in all three (@tbl:verdict). The winner removes about a third of the preset's damage on the 14B and over a quarter on the 1.5B; on the 7B, where a 5 GiB budget already leaves room for a 4-bit preset, the margin is narrow but its interval still lies below 100. The verdict is unchanged when the KL weight $lambda$ is set to 5 or 20 instead of 10.

#figure(
  table(
    columns: 4,
    [], [1.5B @ 1 GiB], [7B @ 5 GiB], [14B @ 5 GiB],
    [Competitor], [`IQ2_S`], [`Q4_K_M`], [`IQ2_XXS`],
    [Score: ours / preset], [23.30 / 32.43], [6.47 / 7.43], [17.48 / 26.25],
    [Relative score], [71.9], [87.1], [66.6],
    [95 % interval], [[67.5, 76.2]], [[75.5, 98.6]], [[61.7, 71.5]],
    [Verdict], [WIN], [WIN], [WIN],
  ),
  kind: table,
  caption: [The verdict on the held-out verdict set (100 = the preset's damage; lower is better).],
) <tbl:verdict>

@fig:verdict shows the three intervals, all wholly below 100, so each win exceeds what sampling luck could explain.

#figure(
  image("plots/verdict.png", width: 100%),
  caption: [The verdict: relative score with its 95 % interval for each contest. All three intervals lie wholly below 100 (the preset's damage), so all three are a WIN.],
) <fig:verdict>

== Accuracy

At the two tight budgets the winner is more accurate than the preset on every one of the five held-out tasks, closing most of the gap to the original (@tbl:acc). The largest recovery is on GSM8K, the step-by-step reasoning task: the `IQ2_S` preset collapses to 4.4 % on the 1.5B while the winner keeps 33.4 % (+29.0 points, against the original's 35.6 %), and on the 14B the winner gains 15.4 points (77.2 % vs. 61.8 %). Overall the winner keeps 93.1 % of the original's accuracy on the 1.5B (preset 73.3 %) and 93.5 % on the 14B (preset 85.2 %). On the 7B both files keep over 99.5 % and differ by at most 1.6 points on any task; there the win comes from staying closer to the original's predictions rather than from accuracy.

#figure(
  table(
    columns: 10,
    [Task], [1.5B ours], [1.5B preset], [1.5B orig.], [7B ours], [7B preset], [7B orig.], [14B ours], [14B preset], [14B orig.],
    [MMLU], [53.8], [45.9], [62.2], [75.4], [74.7], [75.5], [73.6], [69.8], [80.5],
    [HellaSwag], [63.3], [56.1], [67.5], [79.3], [78.8], [79.3], [77.9], [74.8], [83.4],
    [ARC-Challenge], [43.8], [36.8], [46.5], [52.7], [54.1], [53.2], [53.7], [50.9], [61.5],
    [WinoGrande], [62.5], [59.0], [63.9], [71.0], [70.5], [70.8], [75.5], [68.8], [74.4],
    [GSM8K], [33.4], [4.4], [35.6], [75.8], [77.4], [77.2], [77.2], [61.8], [82.8],
    [Mean], [51.4], [40.4], [55.1], [70.9], [71.1], [71.2], [71.5], [65.2], [76.5],
  ),
  kind: table,
  placement: top,
  scope: "parent",
  caption: [Held-out accuracy (%), by task and model.],
) <tbl:acc>

@fig:acc1, @fig:acc7 and @fig:acc14 show the same numbers.

#figure(
  image("plots/accuracy_1.5b@1.png", width: 90%),
  caption: [Held-out accuracy per task, Qwen2.5-1.5B-Instruct.],
) <fig:acc1>

#figure(
  image("plots/accuracy_7b@5.png", width: 90%),
  caption: [Held-out accuracy per task, Qwen2.5-7B-Instruct.],
) <fig:acc7>

#figure(
  image("plots/accuracy_14b@5.png", width: 90%),
  caption: [Held-out accuracy per task, Qwen2.5-14B-Instruct.],
) <fig:acc14>

== Quality

The same pattern holds on the distributional metrics (@tbl:qual): in every contest the winner drifts less from the original than the preset does — lower KL, fewer answer flips and more unchanged top-word predictions. On the 7B the KL is 16 % lower (0.0129 vs. 0.0154), on the 14B 40 % lower and on the 1.5B 61 % lower.

#figure(
  table(
    columns: 7,
    [Metric], [1.5B ours], [1.5B preset], [7B ours], [7B preset], [14B ours], [14B preset],
    [KL divergence], [0.194], [0.494], [0.0129], [0.0154], [0.237], [0.395],
    [Answer flips (%)], [21.4], [27.5], [6.3], [7.3], [15.1], [22.3],
    [Same top word (%)], [79.4], [67.7], [95.1], [94.4], [79.7], [73.7],
  ),
  kind: table,
  placement: top,
  scope: "parent",
  caption: [Held-out quality against the original. Answer flips are the share of the four multiple-choice tasks' questions whose right/wrong answer differs from the original's.],
) <tbl:qual>

#figure(
  image("plots/kl.png", width: 100%),
  caption: [KL divergence against the original (lower is better).],
) <fig:kl>

#figure(
  image("plots/flips.png", width: 100%),
  caption: [Answer flips against the original (lower is better).],
) <fig:flips>

#figure(
  image("plots/sametop.png", width: 100%),
  caption: [Same top-word share (higher is better).],
) <fig:sametop>

== Space and the winner's mix

The search spends each budget in full and puts the bits where they matter. Measured from the files, the 1.5B winner stores its weights at 3.01 bits apiece (81 % smaller than the 16-bit original), the 14B at 2.45 bits (85 % smaller) and the 7B at 5.07 bits (68 % smaller). @fig:model3d, @fig:model3d7 and @fig:model3d14 render this as squares: the light ghost square is the 16-bit original, the filled blue square is the quantised winner, and the dotted squares are the further copies of it that fit in the original's space.

#figure(
  image("plots/layers3d_1.5b.png", width: 50%),
  caption: [Qwen2.5-1.5B-Instruct: the 16-bit original (2.9 GiB) against the quantised winner (0.54 GiB).],
) <fig:model3d>

#figure(
  image("plots/layers3d_7b.png", width: 50%),
  caption: [Qwen2.5-7B-Instruct: the 16-bit original (14.2 GiB) against the quantised winner (4.50 GiB).],
) <fig:model3d7>

#figure(
  image("plots/layers3d_14b.png", width: 50%),
  caption: [Qwen2.5-14B-Instruct: the 16-bit original (27.5 GiB) against the quantised winner (4.22 GiB).],
) <fig:model3d14>

== The winner's multi-bit configuration

The search assigns one quantization type to each matrix, so a winner is a mixed-precision configuration rather than a single type everywhere. @tbl:config lists the final assignment of every matrix of the three winners, read from the files.

#figure(
  table(
    columns: 5,
    [Type], [bits / weight], [1.5B], [7B], [14B],
    [`IQ1_S`], [1.56], [—], [—], [24],
    [`IQ1_M`], [1.75], [—], [—], [6],
    [`IQ2_XXS`], [2.06], [3], [—], [17],
    [`IQ2_XS`], [2.31], [4], [—], [14],
    [`IQ2_S`], [2.50], [1], [1], [277],
    [`Q2_K`], [2.63], [2], [—], [—],
    [`IQ3_XXS`], [3.06], [129], [4], [—],
    [`Q3_K`], [3.44], [32], [—], [—],
    [`IQ4_XS`], [4.25], [11], [16], [—],
    [`Q4_K`], [4.50], [15], [6], [—],
    [`Q5_K`], [5.50], [—], [171], [—],
    [Matrices], [], [197], [198], [338],
  ),
  kind: table,
  caption: [The final mixed-precision configuration of the three winners (matrices per type).],
) <tbl:config>

Each winner has a dominant type close to its budget's average and moves a minority of matrices away from it. The 1.5B winner is mostly `IQ3_XXS` (129 of 197) and raises 58 matrices to 3.4–4.5 bits, paying for them by pushing 10 to 2-bit types. The 14B winner is mostly `IQ2_S` (277 of 338) and pushes 30 matrices down to the 1-bit types `IQ1_S` and `IQ1_M` to keep the rest at 2–2.5 bits. The 7B winner is mostly `Q5_K` (171 of 198), with 27 matrices at 2.5–4.5 bits.

== The winner, distillation and QAT

The winner is chosen on the winner set from the three search finalists, each scored as searched and after distillation. At low bit-widths distillation helped every finalist (@tbl:distill): on the 1.5B the best finalist's score fell from 20.72 to 17.49 and on the 14B from 19.87 to 15.68, with KL falling by about a quarter. At about 5 bits it did the opposite: every 7B finalist got worse, its KL rising about fourfold (0.0148 → 0.0607), so the 7B winner is an undistilled finalist. Choosing the winner on the winner set caught this automatically. No QAT finalist was built in these contests (the stage is optional and off by default).

#figure(
  table(
    columns: 4,
    [Finalist], [1.5B], [7B], [14B],
    [0], [20.72 → *17.49*], [4.81 → 8.27], [19.87 → *15.68*],
    [1], [20.35 → 18.48], [*4.55* → 8.40], [20.30 → 16.49],
    [2], [21.15 → 18.73], [5.34 → 8.95], [21.87 → 17.21],
    [Preset], [31.15], [5.20], [22.56],
  ),
  kind: table,
  caption: [Winner-set score of each finalist before → after distillation, and of the preset (lower is better). The winner is in bold.],
) <tbl:distill>

== How the winners run on the Pi

On the Raspberry Pi 5 every file fits its budget with margin: peak RAM stays 0.2–0.3 GiB under the prediction in every run (@tbl:pi). `ik_llama.cpp` with run-time repacking (`-rtr`) writes fastest on all three models — 11.0 tokens per second on the 1.5B, 2.4 on the 7B and 1.45 on the 14B — while `ik_llama.cpp` without it reads the prompt fastest. The gap is widest on the 14B, whose 1- and 2-bit types run slowly in their stored layout: there `-rtr` writes 1.7 times faster than without it, while on the 7B, mostly `Q5_K`, the three runtimes write within 5 % of each other.

#figure(
  table(
    columns: 6,
    [Contest], [Runtime], [decode (tok/s)], [prefill (tok/s)], [peak RAM (GiB)], [predicted (GiB)],
    [1.5B @ 1 GiB], [ik `-rtr`], [11.01], [58.14], [0.78], [1.00],
    [], [ik], [9.39], [109.59], [0.71], [1.00],
    [], [mainline], [10.50], [15.51], [0.70], [1.00],
    [7B @ 5 GiB], [ik `-rtr`], [2.43], [14.60], [4.71], [5.02],
    [], [ik], [2.31], [14.11], [4.74], [5.02],
    [], [mainline], [2.31], [5.38], [4.75], [5.02],
    [14B @ 5 GiB], [ik `-rtr`], [1.45], [4.42], [4.77], [5.02],
    [], [ik], [0.86], [8.69], [4.77], [5.02],
    [], [mainline], [0.92], [1.25], [4.75], [5.02],
  ),
  kind: table,
  placement: top,
  scope: "parent",
  caption: [Raspberry Pi 5 inference: decode (generation) and prefill (prompt processing) speeds, with peak RAM against the RAM model's prediction (file + KV cache + 0.40 GiB).],
) <tbl:pi>

= Conclusion

We formulated the deployment of large language models on memory-constrained devices as a mixed-precision search: for each (model, RAM budget) contest, assign a quantization type to every matrix so the resulting file fits the budget while losing as little quality as possible. Three choices make that search practical — GPTQ-compensated type copies turn candidate evaluation into a byte copy, a single fitness measure ranks every file on one scale, and disjoint data sets keep training, selection and judgement honest.

Across the three contests reported here, all came out as WIN: the winner cut the competitor preset's damage to a relative 71.9 on the 1.5B @ 1 GiB, 87.1 on the 7B @ 5 GiB and 66.6 on the 14B @ 5 GiB. The gain is largest where the budget is tightest: at 1 GiB and at 5 GiB for the 14B the winner keeps about 93 % of the original's accuracy against the preset's 73 % and 85 %, and recovers 29 and 15 points on step-by-step reasoning (GSM8K), where the low-bit presets fail first. When the budget is roomy, as for the 7B at 5 GiB, a 4-bit preset already keeps over 99.5 % and the search's advantage shrinks to a closer match of the original's predictions; there distillation also stopped helping, which the winner set caught. All the files fit their budgets on a Raspberry Pi 5.

Three directions follow from here. *Training-time quantization* would fold the quantization-aware objective into the model's own training rather than applying QAT only to the winning genome, making still lower bit-widths usable. *Mixture-of-experts* models are a natural next target: since only a few experts are active per token, different experts could be held at very different precisions — the frequently routed experts near-lossless, the rare ones at extreme low precision — turning our per-matrix search into a per-expert one. Finally, *layer-wise loading* (as in AirLLM @airllm), which keeps only the current layer resident and streams the rest from disk, would let the memory budget be spent on activations and the KV cache instead of weights, shrinking the effective footprint far beyond a uniform ~3 bits per weight. Beyond these, the QAT sample count and the solver's ridge strength remain the open hyperparameters to sweep alongside the verdict.


