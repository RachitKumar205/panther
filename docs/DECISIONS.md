# Decision log

Every decision that shapes results, infrastructure or security, with the alternatives and why. Newest
last. A step is written here (status **planned**) *before* it runs, and updated with the outcome after.

Status: **decided** (in effect) · **planned** (agreed, not yet run) · **superseded** (replaced; kept for
history).

---

## Scope and metrics (PLAN.md revisions 1–6, before 2026-11-10)

### D1. Goal: best model + best mixed precision within a RAM budget on an 8 GB Pi 5 — decided
- Find, among Qwen2.5 1.5B/3B/7B/14B Instruct, the best-performing model and its best per-gene format
  assignment that fits a RAM budget R_max, optionally with a decode-speed floor S_min.
- **Why:** the target device is fixed (Pi 5, 8 GB, CPU-only llama.cpp); RAM is the binding constraint.

### D2. Energy removed as a metric — decided
- **Alternatives:** energy budget alongside RAM (original spec); software energy estimates.
- **Why:** no power meter is available, and the team judged that RAM usage covers it (energy is roughly
  proportional to memory traffic for decode-bound inference). Software estimates would add a metric we
  couldn't validate.

### D3. TTFT and load time removed; speed metrics are prefill and decode throughput, per-token latency, end-to-end latency — decided
- **Why:** fewer metrics, each with a clear use. Decode throughput is the only speed number that can
  constrain the search (S_min).

### D4. Pi time budget: ~10 hours total, one unattended priority queue — decided
- **Alternatives:** a 3–4 day Pi campaign (earlier plan revision).
- **Why:** team time constraint. Consequence: the Pi only *confirms* server choices; it does not search.
  See PLAN.md §9 for the queue and what is dropped if time runs out.

### D5. Primary quality endpoint: unweighted mean task accuracy; perplexity as key secondary — decided (team confirmation of the 0.5-point margin still open)
- Tasks: HellaSwag, WinoGrande, ARC-Challenge, MMLU 5-shot, GSM8K 5-shot. Pre-registered decision rule:
  A beats B if the paired-bootstrap 95% CI of the accuracy difference is entirely above +0.5 points;
  otherwise perplexity (≥ 1% relative, CI excluding 0), then RAM, then speed. Holm correction.
- **Why:** accuracy is what users feel; perplexity is far more sensitive and breaks ties. Fixing the rule
  before seeing results prevents choosing the metric that flatters a configuration.

### D6. Genes: attention block and MLP block per layer, plus embedding/head — decided
- Tied models (1.5B, 3B) get one combined `embd_head` gene; untied (7B, 14B) get separate `embd` and `head`.
- **Alternatives:** per-matrix genes (7 per layer); per-layer genes (1 per layer).
- **Why:** per-matrix multiplies the sensitivity cost (penalty table = genes × formats measurements) by
  ~3.5× for a gain the literature suggests is mostly captured by attention-vs-MLP. `ffn_down` as its own
  gene is kept as an optional later split.

### D7. Search: exact multiple-choice knapsack (CP-SAT) with top-k shortlist; GA only as an ablation — decided
- **Alternatives:** GA/NSGA-II as the main search (spec); greedy per-byte (HAWQ-style).
- **Why:** with additive penalties the problem is a knapsack that CP-SAT solves exactly in < 0.1 s for 14B.
  The shortlist and the GA ablation exist because penalties are not exactly additive. Greedy (B4) and
  random search (B5) are baselines.

### D8. Mixed GGUFs assembled by byte-copying tensors from per-format "pure" files — decided
- **Alternatives:** re-run `llama-quantize` with `--tensor-type` patterns for every candidate.
- **Why:** seconds instead of minutes per candidate, and bit-identical to what `llama-quantize` would
  produce for that tensor in that format (tested). Patterns, when used, are anchored `^…$` because
  llama.cpp uses `regex_search` with first-match-wins (`blk.1` would otherwise also match `blk.10`).

---

## Server (the server), 2026-11-10

### D9. Everything containerised, rootless Podman, no host packages — decided
- **Alternatives:** install `g++`, `cmake`, CUDA toolkit, `uv` on the host (needs root, `dnf`);
  user-level conda/micromamba toolchain.
- **Why:** server rules: no software on the host, every project self-contained and removable. Rootless
  means containers run as `svcuser`, never root, so Panther needs no root at all after setup.

### D10. Three host changes, made by the user as root, logged with their undo — decided
1. `/s3/panther` (root) with `/s3/panther/work` (svcuser): `/var` is too small and svcuser can't write `/s3`.
2. SELinux equivalence `/s3/panther/work/containers` = `/var/lib/containers`, so container storage on
   `/s3` gets correct labels.
3. `loginctl enable-linger svcuser`, so jobs survive the SSH session ending.
- Each is written to `/s3/panther/HOST_CHANGES.tsv` with its undo command as it is made; a failure or a
  GitLab health check failing undoes the run. `deploy/server/uninstall.sh` replays the ledger in reverse.
- **Tested:** setup, idempotent re-run, uninstall, rollback on failure and on GitLab breaking, with stubbed
  commands. Testing found and fixed a bug: the rollback trap ran inside subshells while the main script
  continued.

### D11. Panther's container storage separate from svcuser's default storage — decided
- `podman --root /s3/panther/work/containers/storage --runroot /run/user/<uid>/panther`.
- **Why:** uninstall can remove all Panther images without touching svcuser's existing Podman content;
  image layers land on `/s3`, not on `/` or `/var`.

### D12. Base image `nvidia/cuda:12.9.1-devel-ubi9`, pinned by digest; llama.cpp built in the image — decided
- CPU build with `GGML_NATIVE=ON` (built on the server, runs on the server) for final numbers; CUDA build for
  `sm_120` (RTX PRO 6000 Blackwell) for imatrix and screening.
- **Why UBI9:** matches the RHEL host. **Why 12.9:** supports Blackwell; driver 610 supports it.

### D13. PyTorch pinned to the CUDA 12.9 build (2.13.0+cu129) on Linux — decided
- **Found:** PyPI's default Linux wheel was 2.15.0+cu132 (CUDA 13.2), mismatching the image.
- **Alternatives:** move the image to CUDA 13; accept the mismatch (torch bundles its own runtime).
- **Why:** user asked for torch to match the image's CUDA. 2.13.0 is the newest torch published for cu129.
  Set via a `[tool.uv.sources]` index restricted to Linux; the Mac lock entry stays 2.15.0 (unused there:
  torch is only in the `server` extra). Pinning `torch==2.13.0` everywhere would unify the lock — offered,
  not done.
- The converter environment keeps llama.cpp's own pin (2.11.0+cpu); it only reads weights.

### D14. GPU access: SELinux labelling off for GPU containers only, and those get no network — decided by user
- **Found:** SELinux denies `container_t` access to `/dev/nvidia*` (`xserver_misc_device_t`);
  `container_use_devices` is off. Tested: only `--security-opt label=disable` fixes it.
- **Alternatives:** (a) a custom SELinux module allowing only Panther's containers the devices (one more
  host change); (b) `setsebool container_use_devices on` (rejected: opens devices to every container,
  GitLab's included).
- **Why:** no host change. Remaining protection: rootless (keep-id), all capabilities dropped,
  no-new-privileges, read-only root, only repo (ro) and data mounted, seccomp. The cost is losing SELinux
  as a second barrier against container escape. Mitigation:
  GPU containers get `--network none` (and HF offline), so a compromised package in a GPU job can't be
  steered or exfiltrate; downloads happen in separate, fully confined CPU jobs.
- Consequence: model download (`fetch_model.sh`, CPU job) is split from tensor-store building (GPU job).

### D15. Resource caps for jobs: 20 CPUs, 64 GB, lower CPU/IO priority; jobs refuse to start if GitLab isn't 200 — decided
- **Why:** production GitLab shares the 48-thread EPYC and `/s3`. Don't run large jobs while the LLM stack
  (88 GB cap) is up.

---

## Data, 2026-11-10

### D16. Data sources pinned by dataset revision (configs/data.yaml) — decided
- FineWeb-Edu `sample/10BT/000_00000.parquet` (one 2.15 GB file, ~700k docs) for web text; WikiText-2
  raw test; HellaSwag, WinoGrande (xl), ARC-Challenge, MMLU (all), GSM8K (main).

### D17. Web text split by seeded document hash into four disjoint shards — decided
- Each split takes whole documents (128–8192 tokens) from its shard, in hash order, to a token target:
  calibration 300k, sensitivity 100k, selection 150k, test 150k (+ WikiText-2 test 299k = 449k).
- **Why:** disjoint by construction, deterministic, and whole documents keep long-range context for
  perplexity. Token counts use the Qwen2.5 tokenizer (shared by all four sizes).

### D18. Test tasks = full official sets; selection items sampled from non-test splits, minus overlaps — decided
- Selection items whose normalised text matches a test item are dropped (HellaSwag 25, ARC-C 8, MMLU 29).
- **Why:** comparability with published numbers for test; no test leakage into choices.

### D19. MMLU's inherited few-shot/test duplicates kept and recorded — decided
- 5 MMLU test items duplicate items in the standard `dev` few-shot set (an upstream property of MMLU).
- **Alternatives:** drop those test items; replace those exemplars.
- **Why:** keeps the official protocol (comparable with published 5-shot MMLU); affects every configuration
  identically, so paired comparisons are unbiased. Counted in `splits/manifest.json`. Hard checks remain:
  selection ∩ test = ∅ and selection ∩ few-shot = ∅.

---

## Tensor store, 2026-11-10

### D20. Importance matrix: 300k calibration tokens, context 512, BF16 model on GPU — decided
- **Why:** llama.cpp's usual imatrix practice; ~585 chunks. Same calibration text for every model.

### D21. IQ2_XS excluded from the embedding gene only — decided
- **Found:** `llama-quantize` aborted (`GGML_ASSERT(imatrix != NULL)`) forcing the token embedding to
  IQ2_XS: the embedding is a lookup table, so imatrix collection never sees it, and IQ2_XS requires
  imatrix data (`ggml_quantize_requires_imatrix`: IQ2_XXS, IQ2_XS, IQ1_S at the pinned commit). Our config
  had wrongly listed IQ3_XXS and IQ3_S too; corrected.
- **Alternatives:** drop IQ2_XS entirely; quantize the embedding without imatrix (not possible for IQ2_XS).
- **Why:** keeps IQ2_XS for every other gene. In the IQ2_XS source file the embedding holds a Q2_K
  placeholder that the search never copies; verification accounts for it.
- **Outcome (1.5B):** all 12 source files verified, every searchable tensor in its intended format.
- **Extended (2026-11-11, found on 14B):** `llama-imatrix` at the pinned commit does not collect data for the
  untied output tensor by default either, so the 14B IQ2_XS source aborted the same way. IQ2_XS is now
  excluded from the `head` gene as well (a placeholder Q2_K head in that source file). The head is the most
  sensitive gene, so the solver would not choose 2.3 bits for it anyway.

### D22. Every tensor store gets a manifest — decided
- `store/<model>/manifest.json` (written by `scripts/server/store_manifest.py` at the end of every build):
  weights repo@revision, llama.cpp commit, image id, calibration and splits-manifest hashes, imatrix and
  BF16 hashes, per format the file hash, size, read-back result and embedding placeholder, step timings.
- 1.5B was backfilled after its build; its timings are approximate (file times) and say so in the manifest.
- Found while backfilling: git inside the container refused the root-owned llama.cpp checkout ("dubious
  ownership"), so the commit read back empty; fixed with `safe.directory`, and the script now fails if the
  commit can't be read.

### D30. Never retry SSH to the server in a loop; wait with `deploy/server/wait-job.sh` — decided (2026-11-11)
- **What happened:** the SSH control master expired (4 h) while a polling loop on the Mac was waiting
  for the A2 job. The loop kept connecting every 10 s and every attempt failed authentication. the server's
  OpenSSH penalises an address after repeated failures (`PerSourcePenalties`; the server answers
  "Not allowed at this time" before authentication), so the user's own logins with the hardware key were
  refused too.
- **Fix:** `deploy/server/wait-job.sh` checks for a live control master before any connection, polls every
  30 s, and stops at the first SSH failure (exit 255) instead of retrying. All waiting uses it.
- **Recovery:** stop every automated attempt, wait for the penalty to lapse (OpenSSH's default maximum is
  10 minutes), then reopen the control master once.

### D31. Sensitivity chunk count: rule not met, so N = 48 — decided by the pre-registered fallback; follow-up open
- A2 study (PIPELINE.md A2 results): at N = 48, relative SE is 12.6–12.9% for the layer-0 genes and ranking
  agreement is 77%, against the rule's ≤ 5% and ≥ 95%. By the fallback fixed before the run, Phase C uses
  all 48 sensitivity chunks.
- **Consequence:** 48 chunks is the most the current sensitivity split has, and even it isn't precise for
  small, heavy-tailed effects. The 14B penalty table at 48 chunks is ≈ 53 GPU-hours (estimate).
- **Options for the team, not decided:**
  1. Accept N = 48 and the noise: the solver weighs penalties against bytes, and a 13% error on a
     0.007 penalty is small next to the larger penalties that drive its choices. The shortlist and
     refinement (PLAN.md §8) re-measure the chosen configurations on more data.
  2. Enlarge the sensitivity split (e.g. 300k tokens) for more precision. More GPU time per evaluation;
     the splits would be rebuilt (sensitivity only; calibration, selection and test unchanged).
  3. Replace the relative-SE criterion with one tied to the solver (e.g. how often the top-k solution
     changes when penalties are resampled). This would be a new pre-registered rule, set before seeing
     its outcome.
- A2's measured numbers stay as they are regardless; any change is a new decision with its own entry.

---

## Phase A gate (1.5B) — planned

Full step list in [PIPELINE.md](PIPELINE.md#next-phase-a-gate-on-15b--planned). Decisions to make there:

### D27. A1 outcome and how its checks work — decided
- A1 passed (results in PIPELINE.md). Checks: our read-back, llama.cpp's own per-type counts from the
  loader log (needs `-lv 4` at this commit), exact data-size prediction, byte identity per format,
  generation on both builds, perplexity ordering on the sensitivity text.
- Generation uses `llama-cli -st --temp 0` (the CLI is a chat UI at this commit, so the instruct chat
  template applies). Free-text prompts are checked for form (printable, ≥ 3 words, not repetitive); the
  arithmetic prompt is checked by its answer.
- Bugs found and fixed while running: a comment placed before a shell redirect disabled logging, so the
  first report read stale logs; the coherence rule failed the correct answer "42". The final run was
  from an emptied output folder.

### D23. KL reference is the BF16 model, not FP16 — planned
- PLAN.md §2.1 says "KL(FP16 ‖ quantized)". The models are published in BF16, so BF16 is the unrounded
  reference; converting to FP16 would add its own rounding. To be recorded as a deviation in PLAN.md §15.

### D24. Reference-logit storage — decided for 1.5B (references kept until its search is done)
- **Measured** (A2 probe): 303,880 bytes per scored token = 310.9 MB per 2048-token chunk.
  Sensitivity (48 chunks) 14.9 GB + selection (74 chunks) 23.0 GB ≈ 38 GB per model.
- **Proposed:** generate a model's reference files when that model's search starts, keep them until its
  finalists are chosen, then delete them. They are deterministic (same BF16 file, text, commit and
  context) and take minutes to regenerate, so nothing is lost. Peak disk at 14B: ~130 GB tensor store +
  ~38 GB references, against 564 GB free on `/s3`.
- **Alternatives:** keep all four models' references (≈ 150 GB); store top-k logits only (llama.cpp
  has no such mode; would need our own code and loses the exact full-vocabulary KL).
- **Final-test KL** (PLAN.md §2.1 lists KL among final metrics): the full test text would need 219
  chunks ≈ 68 GB per model. Proposed: KL on the test split uses the web-text part only (73 chunks,
  ≈ 23 GB), generated for finalist models only, deleted after. Perplexity still uses all 449k tokens.
- How many sensitivity chunks Phase C actually reads is set by the A2 chunk-count study (rule in
  PIPELINE.md A2, step 3).

### D28. KL runs always repeat the base run's context and batch — decided
- `--kl-divergence` must be given `-c 2048 -b 2048` like the base run; without it llama.cpp fails to
  decode (found in the A2 probe). The KL wrapper will take these from the reference manifest, never
  from defaults.

### D29. One CUDA-built reference serves CPU-build evaluations — decided (from the A2 probe)
- CPU-vs-CUDA BF16 on the same reference: mean KL 0.00005, top-1 99.56%, PPL ratio 1.0002, about
  1,000× below uniform Q4_K's KL (0.063). So references are built once on the GPU; final CPU-build
  evaluations compare against them. The measured build floor is reported alongside final KL numbers.

### D32. Per-chunk NLL from `--ppl-output-type 1` — planned (A3)
- The perplexity bootstrap in the decision rule (D5) resamples chunks, so it needs each chunk's NLL.
  llama.cpp prints only running averages: perplexity to 4 decimals by default, or mean NLL to 6
  decimals with `--ppl-output-type 1`. With equal tokens per chunk, per-chunk NLL is recovered as
  `i·m_i − (i−1)·m_(i−1)`, error ≤ (2i − 1) × 5×10⁻⁷ ≈ 2×10⁻⁴ at 219 chunks, versus typical per-chunk
  differences between configurations of 0.01–0.05. The recovered values must reproduce the final
  perplexity within 0.01% or the run is rejected.
- **Alternatives:** one run per chunk (exact, but loads the model 219 times per evaluation); patching
  llama.cpp to print more (a change to a pinned dependency).

### D33. One JSON record per evaluation in `data/results/<model>/` — planned (A3)
- Each record states what was evaluated, how, the metrics and the per-chunk values, so any table or
  decision can be recomputed from records alone. Records are append-only: a re-run gets a new record.

### D34. The Phase A gate never reads the test split — planned (A3)
- PLAN.md's test split is "touched once", for finalists only (§3). The gate proves that the
  perplexity and KL metrics *run* using the selection and sensitivity texts; test-split numbers first
  appear in Phase D.

### D35. CPU vs CUDA gap for quantized models — check run 2026-11-11: failed, handled by D36
- **Found:** A1's KL is 1.3% higher on the CPU build than on CUDA (0.15629 vs 0.15425), perplexity
  0.12% higher; BF16 is identical on both. A3's check 3 compared this with the BF16 build floor
  (0.00005), the wrong yardstick for quantized kernels, so it failed. The check is left failed.
- **Cause (to confirm):** the CPU build quantizes activations to 8 bits for quantized matrix products;
  the CUDA kernels differ. The Pi runs the same CPU code path, so the CPU build is the faithful one.
- **Why it matters:** Phase C screens on CUDA (the penalty table would take ~10× longer on the CPU). That
  is fine if the gap is consistent, i.e. CUDA ranks configurations the same way the CPU build does.
- **Proposed follow-up (pass rule fixed before running):** KL on the sensitivity text for all 12
  uniform formats on the CPU build (~35 CPU-minutes), compared with the CUDA sweep. Pass if (a) both
  builds rank the 12 formats identically, ties allowed only where two formats are within 2% of each
  other on CUDA, and (b) the CPU − CUDA relative difference lies within ±5% for every format.
  - Pass: keep CUDA screening; report final numbers on the CPU build (as PLAN §2.1 already says) and
    state the measured gap.
  - Fail: screening must use the CPU build or a measured per-format correction; a new decision with
    its cost.
- **Outcome (2026-11-11): FAIL.** (a) ranking: identical on both builds, no swaps. (b) relative
  difference: fails, max +123% (Q6_K). CPU − CUDA in absolute KL, sensitivity text, uniform formats:

  | Group | Format: CPU − CUDA |
  |---|---|
  | A32 | Q8_0 +0.00001 · IQ4_NL +0.00002 · Q4_0 −0.00012 |
  | A256 | Q6_K +0.00714 · Q5_K +0.00756 · IQ4_XS +0.00764 · Q4_K +0.00774 · IQ3_S +0.00847 · Q3_K +0.00879 · IQ3_XXS +0.00916 · Q2_K +0.01033 · IQ2_XS +0.01213 |

  The predictions made before the last four formats finished all held (IQ4_XS, IQ3_S, IQ3_XXS rose by
  0.008–0.009; IQ4_NL unchanged). Handled by D36. Report: `data/gate/d35/report.json`.

### D36. CUDA screening plus a measured CPU activation correction; decisive comparisons on the CPU build — decided by user 2026-11-11; sampled check run: FAIL (see outcome)
- **Why:** D35's CPU sweep shows the CPU build adds a roughly *constant* KL (+0.007 to +0.010) to every
  format whose CPU kernel rounds activations to `Q8_K`, and ≈ 0 to the others. CUDA alone would overrate
  high-bit K-quants for the Pi; CPU-only screening of 14B would take 400+ hours.
- **Mechanism (from `ggml-cpu.c` at the pinned commit, `vec_dot_type`):**
  - group **A32** (activations to `Q8_0`, one scale per 32 values): Q8_0, Q4_0, IQ4_NL;
  - group **A256** (activations to `Q8_K`, one scale per 256 values): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
    IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS.
  The coarser activation rounding costs about the same however precise the weights are, so it swamps
  Q6_K's small weight error but is noise next to Q2_K's. The Pi runs the same CPU code.
- **Model:** CPU KL(config) ≈ CUDA KL(config) + Σ over genes in an A256 format of a_g, where a_g is gene
  g's activation-rounding cost. For the penalty table this means a gene's CUDA penalty is corrected only
  when the change moves it between groups: penalty_cpu(g, f) = penalty_cuda(g, f) + a_g × ([f ∈ A256] −
  [base format of g ∈ A256]).
- **Measuring a_g (1.5B, all 57 genes):** base B = uniform Q8_0 (all A32). For each gene, B_g = B with
  that gene in Q6_K (A256 with the smallest weight error). a_g = [KL_cpu(B_g) − KL_cpu(B)] −
  [KL_cuda(B_g) − KL_cuda(B)], on the 48 sensitivity chunks. Each B_g is assembled, evaluated on both
  builds, then deleted (57 × 1.5 GB would not fit comfortably). ≈ 3 CPU-hours.
- **Checks, fixed before running:**
  - **C1 (additivity):** Σ a_g is within ±25% of the measured uniform-Q6_K gap, 0.007145 (D35:
    CPU 0.012955 − CUDA 0.005810).
  - **C2 (prediction):** for a validation set of 5 mixed configurations (A1 and 4 seeded random
    assignments over each gene's allowed formats), predicted CPU KL = CUDA KL + Σ a_g (scaled per
    format, see the amendment below) must satisfy
    |predicted − measured CPU| ≤ 0.25 × |CUDA − measured CPU| or ≤ 0.0005, for every configuration;
    i.e. the correction removes at least 75% of the gap.
  - **C3 (structure, informative):** how a_g varies with layer and with attention vs MLP, and the error
    of predicting every gene from every 4th layer by interpolation. Decides whether 14B can be measured
    on a sample of layers; that is a separate decision.
  - Pass: C1 and C2. If either fails, the additive model is wrong and the next step is a new decision.
- **Change before any gene was measured (2026-11-11, user):** sampled run instead of all 57 genes.
  14B will have to use a layer sample anyway (≈ 40 CPU-hours for all 98 genes), so the 1.5B pilot tests
  the sampled method directly:
  - measured: `embd_head` and attention + MLP of layers 0, 4, 8, 12, 16, 20, 24, 27 (17 genes,
    ≈ 1 hour instead of ≈ 3.3);
  - every other layer's a_g is linearly interpolated per kind (attention, MLP) from the measured layers;
  - C1 and C2 are applied unchanged, using the interpolated values for unmeasured genes;
  - C3 reports the measured values by layer and kind (its interpolation-error part needs all genes and
    is dropped);
  - if C1 or C2 fails, the remaining 40 genes are measured (≈ 2.3 h) before any other conclusion.
  At the time of the change only the base configuration's CUDA run had finished.
- **Also part of D36 (Phase C design):** the refinement step and the shortlist are scored on the CPU
  build, so every comparison that picks a configuration uses the Pi's arithmetic; final numbers were
  already CPU-only (PLAN §2.1).
- **Per-format scaling (amendment, 2026-11-11, user; before any validation result existed):** the
  whole-model CPU − CUDA gap grows as formats get coarser (D35: Q6_K 0.0071 … Q2_K 0.0103, IQ2_XS 0.0121),
  so a single a_g measured with Q6_K undercharges low-bit formats. The correction becomes
  a_g(f) = a_g × r_f with r_f = gap(f) / gap(Q6_K) from the D35 uniform sweep, and r_f = 0 for the A32
  formats (Q8_0, Q4_0, IQ4_NL). Ratios (from D35's records): Q6_K 1.00, Q5_K 1.06, IQ4_XS 1.07,
  Q4_K 1.08, IQ3_S 1.19, Q3_K 1.23, IQ3_XXS 1.28, Q2_K 1.45, IQ2_XS 1.70. No extra measurement.
  - **Assumes** each gene's share of the extra error is the same in every A256 format.
  - **C2 uses the scaled prediction** (primary); the unscaled prediction is reported alongside.
    C1 is unchanged (it compares Q6_K costs with the Q6_K gap, where r = 1).
  - When the amendment was made, 5 of the 17 sampled genes and no validation configuration had been
    measured.
- **Outcome of the sampled run (2026-11-11): FAIL, as pre-registered.** Report:
  `data/gate/d36/report.json`.
  - **C1 passes:** Σ a_g = 0.007663 vs the uniform-Q6_K gap 0.007145 (+7.2%, limit ±25%).
  - **C2 fails** with the scaled prediction (the amendment's primary): A1 and val0 miss the limit.

    | Config | Genes in A256 | CUDA KL | CPU KL | Uncorrected error | Scaled: error (gap removed) | Unscaled: error (gap removed) |
    |---|---|---|---|---|---|---|
    | A1 | 43 | 0.15425 | 0.15629 | 0.00203 | 0.00059 (71%) ❌ | 0.00008 (96%) |
    | val0 | 42 | 0.20843 | 0.21077 | 0.00234 | 0.00135 (42%) ❌ | 0.00034 (86%) |
    | val1 | 46 | 0.38108 | 0.38984 | 0.00876 | 0.00017 (98%) | 0.00142 (84%) |
    | val2 | 45 | 0.18691 | 0.18955 | 0.00264 | 0.00046 (83%) | 0.00001 (100%) |
    | val3 | 39 | 0.33544 | 0.34319 | 0.00775 | 0.00114 (85%) | 0.00066 (91%) |

  - The **unscaled** prediction (D36's original C2 formula, before the amendment) would pass on all
    five, removing 84–100% of the gap. This is an observation after the fact, not a pass: the
    amendment made the scaled version primary before results existed, and the check is left failed.
  - **What the failure says:** the per-format scaling assumption (each gene's share of the extra error
    is the same in every format) is wrong in at least some configurations; it overcorrects A1 and val0.
    The additive per-gene model itself is supported by C1 and by the unscaled column.
  - **C3, measured a_g (Q6_K, CPU − CUDA):** `embd_head` 0.004444 (58% of the total); `mlp.27`
    0.000762 (the last layer's MLP, 10%); every other sampled gene 0.000008–0.000083 (attention:
    0.000083 at layer 0, then 0.000008–0.000029; MLP: 0.000021–0.000057 for layers 0–24). Nearly all
    of the extra CPU error comes from the output head and the last MLP.
  - Next step: D37 (proposed, needs approval).
- **GA ablation (added 2026-11-11):** the GA measures assembled candidates instead of summing prices.
  Its fitness evaluations run on the CUDA build and get the same correction (CUDA KL + Σ a_g over the
  candidate's genes in an A256 format); its best candidates are then re-measured on the CPU build, exactly
  like the solver's shortlist. Both methods are thus judged on the same scale.

### D37. After D36's failure: revert to the unscaled correction and confirm it on fresh configurations — decided: PASS 2026-11-11
- **Situation:** D36's pre-registered fallback says "if C1 or C2 fails, measure the remaining 40 genes".
  That fallback targets an interpolation problem, but the failure is the scaling assumption: the same
  sampled a_g values pass every configuration when not scaled. Measuring 40 more genes (≈ 2.3 h) would
  not test the thing that failed.
- **Proposal:**
  1. Drop per-format scaling; the correction is D36's original rule, a_g added once per gene in an A256
     format.
  2. Because choosing it now is informed by the results above, confirm it on **4 new seeded random
     configurations** never measured before (≈ 15 min), with the original C2 rule unchanged:
     |predicted − CPU| ≤ 0.25 × |CUDA − CPU| or ≤ 0.0005, for every one.
  3. Pass: adopt it for Phase C. Fail: measure the remaining 40 genes, then a new decision.
- **Run details (fixed before running):** `scripts/server/d37.py`; the 4 configurations use seed
  20261112 (D36 used 20261111), each gene drawn uniformly from its allowed formats; the a_g values are
  D36's 17 measured genes with linear interpolation, unchanged; labels `d37:val:0..3`.
- **Outcome: PASS** (all 4 within the rule). Report: `data/gate/d37/report.json`.

  | Config | Genes in A256 | Head | mlp.27 | CUDA KL | CPU KL | Predicted CPU | Error | Gap removed |
  |---|---|---|---|---|---|---|---|---|
  | val0 | 48 | Q3_K | Q5_K | 0.27068 | 0.27884 | 0.27754 | 0.00129 | 84% |
  | val1 | 42 | IQ3_XXS | Q8_0 | 0.36808 | 0.37508 | 0.37454 | 0.00054 | 92% |
  | val2 | 44 | Q4_K | IQ3_S | 0.19242 | 0.20068 | 0.19973 | 0.00096 | 88% |
  | val3 | 39 | Q2_K | IQ3_XXS | 0.76338 | 0.77143 | 0.77048 | 0.00096 | 88% |

  - **Adopted for Phase C:** penalty_cpu(g, f) = penalty_cuda(g, f) + a_g × ([f ∈ A256] − [base_g ∈ A256]),
    with a_g measured once per gene in Q6_K (no per-format scaling).
  - **Residual, noted for later:** the prediction is below the CPU value in all four (by 0.0005–0.0013,
    8–16% of the gap). Coarser formats do cost a little more on the CPU, just not in proportion per gene
    as D36's scaling assumed. Small next to these configurations' KL (0.19–0.77); the CPU-build
    refinement and shortlist (D36) measure directly and are unaffected.
  - **For 7B and 14B:** the same sampled method; the 1.5B result that most of the cost sits in the output
    head and the last MLP is a hypothesis to check there, not an assumption.
- **Alternative:** follow D36's fallback literally (measure 40 genes, re-analyse with scaling); expected
  to fail the same way, at 2.3 h.

### D25. Task scoring follows llama.cpp's modes; differences from lm-eval documented and measured — decided (source read 2026-11-11)
- Read in `tools/perplexity/perplexity.cpp` at the pinned commit:
  - **HellaSwag / multiple choice:** each answer scored by mean log-prob per token from the first token
    after the shared prefix; highest wins. lm-eval `acc_norm` divides by byte length instead.
  - **WinoGrande:** `first + option + second` per option; score = mean log-prob of the tokens after the
    option (when both continuations have more than 3 tokens), *excluding the last token*. lm-eval's
    partial scoring sums all continuation tokens.
  - **MMLU** with single-letter answers: one token per answer, identical to lm-eval's `acc`.
  - `--hellaswag` always shuffles (fixed seed 1); `--winogrande` and `--multiple-choice` keep file order
    when all items are used. Defaults use subsets (HellaSwag 400), so the count is always set explicitly.
- **Why accept the differences:** every configuration is scored identically, so comparisons are unbiased;
  absolute numbers are not directly comparable with lm-eval leaderboards. A4.4 measures the size of each
  difference on the same items.

### D38. HellaSwag, ARC-C and MMLU via `--multiple-choice`; WinoGrande via `--winogrande`; prompts per lm-eval — planned (A4)
- `--multiple-choice` uses the same scoring as `--hellaswag` but keeps file order, so each result maps to
  a question id (the decision rule's bootstrap pairs questions across configurations). Prompt layouts in
  PIPELINE.md A4. Task runs use `-c 4096 -np 5`: llama.cpp evaluates each answer as its own sequence and
  stops (exit code 0) at the first question with more answers than `-np` (found in A4: ARC-C question 460
  has 5 choices; the default is 4). Every task run uses the same `-np 5`, so all configurations are
  batched identically.
- WinoGrande CSVs end with two newlines: `-f` strips one trailing newline from the file, and the CSV loader
  then discards a final line that reaches end-of-file, silently scoring one question fewer (found in A4:
  999 of 1,000). The parser port in `panther.quality.tasks` models both steps, and a test checks it.

### D39. Independent check of task scoring in transformers — planned (A4)
- 200 seeded selection items per task, same prompts, llama.cpp's scoring rule reimplemented in Python on
  the BF16 model; per-question agreement with llama.cpp ≥ 95% per task. Catches converter or prompt errors
  that a self-consistent pipeline would not.

### D40. Cost of final task accuracy on the CPU build — measured in A4; decision needed before Phase D
- MMLU test alone is 14,042 five-shot prompts (~11M tokens); on the CPU build this is hours per
  configuration for 1.5B and roughly 10× that for 14B. A4 records CPU timings; Phase D's plan (all tasks on
  CPU, CUDA with a CPU subset, or another design) is decided from the measured numbers before Phase D.
- **Change (2026-11-11, user):** A4's full CPU stage (BF16 and A1 on every selection question, ≈ 90 min) was
  stopped before finishing anything and replaced by a timing sample: CPU build, A1, 100 questions per task
  (llama.cpp's own seeded subset), timing only. Script: `scripts/server/time_tasks_cpu.sh`.
- **Measured (A1, 1.5B, 20 threads, per question):** HellaSwag 0.30 s, ARC-C 0.14 s, MMLU 1.11 s,
  WinoGrande 0.09 s. Full test split per configuration: ≈ 5.3 CPU-hours for 1.5B (MMLU 4.3 h); expect
  roughly 10× for 14B. On CUDA the full test for 1.5B is a few minutes.
- **Consequence:** PLAN §2.1's "final task accuracy on the CPU build" for top-2 per model plus baselines
  is not feasible for 7B/14B in the time available. Options for Phase D, to be decided:
  1. Final task accuracy on CUDA for every finalist; CPU build only for perplexity and KL (cheap), plus
     a CPU task check on a fixed subset (e.g. 200 questions per task) to report the CPU/CUDA difference.
  2. CPU build on a fixed test subset (e.g. 2,000 MMLU questions) for every finalist; wider CIs.
  3. Full CPU runs for 1.5B/3B only; CUDA for 7B/14B.

### D26. Generation-sanity prompts — decided (A6)
- 50 prompts, 10 per task, seeded draw from the selection items of HellaSwag, ARC-C, MMLU, WinoGrande and
  GSM8K, each wrapped in a short instruction and sent through the chat template; greedy, 128 tokens.
  Stored in `splits/gen_prompts.json`. Details and pass rules in PIPELINE.md A6.

### D41. GSM8K: our own runner reproducing lm-eval 0.4.13's `gsm8k` task — decided (A5)
- PLAN §2.1 named lm-evaluation-harness with `local-completions`. lm-eval downloads the dataset from the
  Hub, and GPU containers have no network (D14). The task definition (prompt layout, stop strings, both
  extraction regexes, exact-match normalisation) is copied from the pinned lm-eval 0.4.13 `gsm8k.yaml` and
  run against `llama-server` over localhost, on our split's items and fixed exemplars.
- **Primary metric:** strict-match (`#### N`), fixed before any run; flexible-extract reported alongside.

### D42. Proof of concept first: 7B vs 14B at R_max = 7.0 GiB, two independent lanes — decided by user 2026-11-11
- **Why:** a first end-to-end answer in hours instead of days. At 7.0 GiB, 1.5B and 3B fit unquantized, so
  the real contest is 7B (≈ 7.2 bits per weight available) against 14B (≈ 3.4).
- **Budget:** R_max = 7.0 GiB at 4,096 context, 0.40 GiB overhead (`configs/budgets.yaml`); an assumption until
  the Pi measures its real headroom (7.45 GiB physical, minus the OS).
- **Two lanes run independently on the server,** each a chain of jobs that starts the next step itself; 7B gets
  more CPU and memory and is not blocked by 14B. The two meet only for the final comparison.
- **Shortcuts (all labelled PoC; none of its numbers are final):**

  | Item | Full plan | PoC |
  |---|---|---|
  | Formats | 12 | 7B: Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ3_S · 14B: Q6_K, Q4_K, IQ4_XS, IQ3_S, IQ3_XXS, Q2_K, IQ2_XS |
  | Sensitivity text | 48 chunks | 12 chunks |
  | CPU correction (D37) | sampled genes | 7B: output head and last MLP only; 14B: none needed (every format in its menu rounds activations the same way, so the correction never changes a choice) |
  | Shortlist | top 20, CPU build | top 5, CUDA + correction, on selection data |
  | Final scoring | full test split | fixed test subset; **runs only after the user's explicit OK** (test data, D34) |
  | Pi | 10-hour queue | none; RAM predicted exactly |

- **Penalty base:** the menu format closest to the budget everywhere (7B uniform Q6_K, 14B uniform IQ3_XXS);
  each gene in each other format is one KL measurement on CUDA.
- **Expected precision:** configurations within a few percent of the full pipeline's on KL; the 7B-vs-14B
  verdict is reliable only if they differ by about 2 benchmark points or more, otherwise "too close to call".
- **Test-subset approval (user, 2026-11-11):** the user approved the fixed PoC test subset for P7. The full
  test split stays reserved for Phase D. Subset items are recorded by id in `splits/llama_tasks/manifest.json`
  (`poc_test/*`); the full Phase D run will report results with and without them.
- **Unattended finish (user, 2026-11-11):** the user asked for the PoC to run to completion overnight and be ready
  for the Pi in the morning. A final chain on the server waits for both lanes, runs P7, and packages the Pi bundle;
  the Pi side (Phase E, PoC version) is in PIPELINE.md.


### D43. Unattended runs are guarded: Panther pauses itself if the co-located service degrades — decided 2026-11-11
- **Why:** the PoC runs overnight with nobody watching; jobs check service health only when they *start*.
- **What:** `deploy/server/guard.sh` checks `PANTHER_HEALTH_URL` every 60 s. After 3 consecutive bad checks (not
  HTTP 200, or slower than 3 s) it runs `podman pause` on every running `panther-*` container (frozen, not
  killed); after 5 consecutive good minutes it unpauses them. It acts only on Panther's own containers, in
  Panther's own container storage, and logs to `data/logs/guard.log`.
- Measured when it was started (01:47 UTC): service responses 45–115 ms, CPU 55% idle, I/O wait 4%.

---

### D44. Quick Pi check: one file, RAM and one speed measurement (~15 min) — decided by user 2026-11-11
- **Why:** the full Phase E queue takes 3–4 h, mostly repeats, a thread sweep and secondary workloads. The two
  answers that matter for the proof of concept are: does the recommended file fit in 7.0 GiB on the real Pi, and
  how fast is it.
- **What:** `python3 pi/panther_pi.py run --quick`:
  1. checks the sha256 of the recommended file (only the files present are checked);
  2. loads it once at 4 threads (the Pi 5 has 4 cores; the thread sweep is skipped) and runs W_ref 256/64,
     1 warm-up + 2 measured, with the same RAM, temperature, throttling and tensor-type checks as E5;
  3. if it is invalid (over 7.0 GiB, fails to load, MemAvailable floor) it does the same for the fallback,
     the other model's file the verdict prefers (here `qwen2.5-7b-poc-uniform.gguf`), if that file was copied.
  `pi/fetch_bundle.sh PI_HOST quick` copies only the manifest, the agreement text and the recommended file
  (~6 GB instead of ~25 GB).
- **Not covered:** the thread sweep (E4), all four files (E5), server agreement (E6) and other workloads (E7).
  The full queue stays available and is resumable on top of the quick run (different run keys).

---

### D45. The rewrite runs the search.yaml table with a minimal number of KL runs (~390) — decided by user 2026-11-11
- **Why:** the rewrite plan (TODO.md, by icingtea) as written needs 1,800–7,000 KL runs (1–2 per matrix, a
  separate base per model × budget, up to 3 re-measure rounds). The user: keep KL runs to a minimum and stick to
  the table; GPU hours themselves are not a constraint.
- **Contests (unchanged, `configs/search.yaml`):** searched 1.5B@1, 7B@3, 7B@5, 14B@5, 14B@7, 32B@7 GiB;
  references 3B@3 and 7B@7 (uniform, no search). 32B stays in (user).
- **What changes against the plan:**
  1. **Sensitivity per block, allocation per matrix.** One KL run per block (attention and MLP of each layer, plus
     embedding/head): move that block from the base to a probe format, measure the KL increase ΔKL_b. Each
     matrix m in block b then gets `penalty(m, f) = s_b × (e(m, f) − e(m, base))` with
     `s_b = ΔKL_b / Σ_{m∈b} (e(m, probe) − e(m, base))`, where e is the importance-weighted dequantization error
     computed offline from the weights. The solver still picks one format per matrix. This is the form Phase 0.3
     can check against the PoC's measured 14B table (which is per block).
  2. **One sensitivity sweep per model,** shared by its budgets (7B and 14B appear twice).
  3. **No re-measure loop by default.** The top 3 per contest are measured; only if their measured KL is far from
     the prediction are blocks re-measured.
  4. **Scoring per contest:** top 3 + at most 3 same-model references (largest uniform that fits, best stock
     llama.cpp preset that fits, uniform with head and V raised, padded to the budget).
  5. **Cross-model references are compared on accuracy only** (KL against different originals is not comparable).
- **Count:** Phase 0 ~7, sensitivity 343 (1.5B 57, 7B 58, 14B 98, 32B 130), scoring 36: **~390 KL runs, ~1.5 h**.
- **Kept, against the plan's Phase 1 (user):** `deploy/server/` (anton's rules require containers, the ledger and
  the guard); `docs/DECISIONS.md` and `docs/PIPELINE.md` (document before doing). Stale markdown is deleted as
  the rewrite replaces it.
- **Gate:** Phase 0.1 (beat stock presets at the same RAM) and 0.3 (the penalty model predicts the measured table)
  run first; if either fails, stop and report before building the rest.

---

### D46. How the rewrite runs the table (implementation choices) — decided 2026-11-11, within D45
- **Per-matrix search with the existing tools:** a "matrix inventory" makes every weight matrix its own gene
  (`attn_q.L`, …, `ffn_down.L`, `embd`, `head` / `embd_head`), so the assembler, read-back check, RAM model and
  CP-SAT solver are reused unchanged. Blocks (`attn.L`, `mlp.L`, `embd`, `head`) are only used for sensitivity.
- **Format menu per model** (from `configs/search.yaml`'s 12 formats): those with bits/weight between the model's
  lowest budget average − 1.5 and its highest budget average + 2.5, so stores only hold formats that can be
  chosen: 1.5B IQ1_S–Q5_K, 7B IQ1_S–Q6_K, 14B IQ1_S–Q5_K, 32B IQ1_S–IQ3_S.
- **Importance matrix:** the existing ones (1.5B, 7B, 14B) are reused, so the PoC store files stay valid and only
  missing formats are built; their output head keeps the PoC rule (no imatrix-only formats: IQ1_S, IQ1_M,
  IQ2_XXS, IQ2_XS), which costs nothing at their bit widths. 32B gets a new one with `--process-output`
  (calibration text, `-c 512`), so its head may use every format: without it, 32B at its smallest is ~7.08 GiB
  and could not enter the 7 GiB contest.
- **RAM model:** as `configs/search.yaml`: KV cache q8_0 (34 bytes per 32 values) at 4,096 context, overhead
  0.40 GiB (placeholder until the Pi measures it, step 0.2).
- **Sensitivity base per model:** the menu format whose uniform file is closest to (not above) the lower budget's
  average bits/weight; genes that cannot use it take the next format up. Probe format: chosen by Phase 0.3.
- **Scoring:** the top 3 and the same-model references get KL on the selection text; the contest winner is the
  lowest KL of the top 3 (selection data only). Winner, references and cross-model references then get the PoC
  verdict's evaluation (test subset: 4 tasks, GSM8K 300, perplexity) and paired-bootstrap comparisons.
- **References:** largest uniform menu format that fits; the largest stock llama.cpp preset that fits (built
  with `llama-quantize` and the same importance matrix); uniform with head and every `attn_v` raised (to Q6_K,
  or the highest menu format), padded with whole layers raised one format, up to the budget. Cross-model
  references (3B@3, 7B@7): uniform Q6_K, accuracy only; 3B is downloaded for this.
- **Disk:** Hugging Face weights are deleted after conversion to BF16 GGUF (re-downloadable at the pinned
  revision).

---

### D47. Sensitivity is adaptive: one probe per block, then measure around the solution (3 rounds) — decided by user 2026-11-11
- **Why:** Phase 0.3 failed its gate. On the PoC's measured 14B table, one probe per block scaled by the offline
  error ranks penalties well (Spearman 0.87) but is ~45% off in value; the 7 GiB solution it picks has 11.4% worse
  selection KL (0.1039 vs 0.0932). Offline tests on the same table (no KL runs): interpolating from 3 measured
  formats per block loses ~7% of the achievable gain; **measuring around the solution for 3 rounds comes within
  ~1.2% of the fully measured optimum with 63% of the measurements.** The user chose this (~1,300–1,600 KL runs,
  ~3–4 h) over the one-probe plan (~350) and full measurement (~2,750).
- **What:** per model, at block level (attention / MLP per layer, embedding, head):
  1. measure every block moved from the base to the probe: the highest menu format at most 3 bits/weight above
     the base (the up-probe was best in 0.3);
  2. predict unmeasured (block, format) values by piecewise-linear interpolation in the block's offline error
     through the measured points and (base, 0), extrapolating with the edge segment's slope;
  3. solve each of the model's budgets (per matrix); for every block, measure the formats its matrices use and
     their menu neighbours (±1) that are not yet measured; repeat, 3 rounds; then solve the top 3.
  Per matrix: `penalty(m, f) = r_b(f) × (e(m, f) − e(m, base))` with `r_b(f) = ΔKL_b(f) / (E_b(f) − E_b(base))`,
  the block's measured or interpolated ratio, so a block with all matrices at f costs exactly ΔKL_b(f).
- Supersedes D45 item 1–3 (one probe, no re-measure loop); the rest of D45/D46 stands.

---

### D48. Fast path: 7 GiB first, one re-measure round, more parallel KL runs — decided by user 2026-11-11
- **Why:** results were too slow; the user asked to narrow the work and parallelise. Format files (CPU-only
  quantization) were the bottleneck while the GPU sat mostly idle.
- **What:** `active_budgets: [7, 1]` in `configs/search.yaml`: only these budgets are searched now (7 GiB is the Pi's
  real budget; 1 GiB because 1.5B is ready and gives an early end-to-end result). 3 and 5 GiB, and 7B's search,
  wait. Menus and sensitivity bases follow the active budgets (14B: base IQ3_S, probe Q5_K; needs only IQ2_S
  and Q5_K added). Two re-measure rounds instead of three. **Correction (same day):** first set to one round on a misread of
  the offline test; there one round is still ~11% off the fully measured optimum, two rounds ~2.5%, three ~1.2%,
  and the second round costs only ~15% more runs (14B: 311 → 360). Up to 10 KL runs at once (GPU memory check still applies). 32B references are made
  right after its BF16 file, before its formats, while the GPU is free. Evaluation per budget: each model's
  winner, its best same-model reference by selection KL, and the cross-model reference.
- **1.5B sensitivity text:** its phase-A reference was 48 chunks (~98k tokens), 4× the other models' 12; its sweep
  restarts on a 12-chunk reference like the others (the 48-chunk file is kept as `sensitivity-48chunks.kld`).

---

### D49. One standard: a fixed solution list per budget, each genotype assembled once and fully tested — decided by user 2026-11-11
- **Why:** the user found the run patchworked and slow: files were rebuilt for every measurement, candidates did not
  have to fill the budget, tests differed per step, and there was a lot of one-off code.
- **What:** `docs/STANDARD.md` is the procedure from now on. The solution space for a budget is a precomputed list of
  genotypes inside RAM −1% … +0.5% of the budget (solver top 6 per model, ≥5% apart, plus a padded-uniform
  baseline). Each is assembled once and gets every quality test on the GPU; the best by accuracy wins and goes to
  the Pi. Part-cost measurements (step 2) stay as instruments: offline tests on the 14B data showed choosing with
  no measurements captures ~75% of the achievable gain (about llama.cpp-preset level), and learning part costs
  from measured whole genotypes only 64–73% even with 200 of them.
- **Removed:** the contest/eval split, llama.cpp presets and head/V references (the Phase 0 gate already compared
  against presets), the CPU/GPU correction scripts, Phase A gate scripts, `phase0.py`, greedy/random baselines.

---

### D50. One command per budget; each model goes straight to its winner and the Pi; reflink copies for part costs — decided by user 2026-11-11
- **Why:** the user: the pipeline must be code, not hand-written job lists; a model whose measurements are done must
  go to the solver and its final picks immediately; building a whole model file for every measurement is wasteful.
- **What:** `deploy/server/run-budget.sh <B>` runs one lane per model (quantized copies → offline error → reference
  → part costs → solver list → tests → winner → Pi bundle) and a final cross-model comparison; it is idempotent and
  never rewrites a running lane. `pi/relay.sh <B> <size>` (Mac) sends a winner to the Pi as soon as it is bundled.
  Part-cost files are reflink copies of the model's starting-type file with the changed part appended, used only if
  a tensor-by-tensor comparison with a normally built file matches. Explanation and numbers: `docs/STANDARD.md`.

---

## Open

- Team value for R_max (`configs/budgets.yaml`), required before Phase C.
- Team confirmation of task accuracy as primary endpoint and the 0.5-point margin (D5).
- Whether to keep one Pi energy check (D2 removed energy; question raised, not answered).
- Whether to pin `torch==2.13.0` on every platform (D13).
- How to handle sensitivity precision for Phase C (D31: accept N = 48, enlarge the split, or a
  solver-based rule).
- How 14B measures its per-gene activation correction (D36 C3 decides whether a layer sample suffices).
- How Phase D measures final task accuracy given the CPU cost (D40).
