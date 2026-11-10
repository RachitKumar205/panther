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

### D25. Task scoring definitions follow llama.cpp's task modes, checked against source first — planned
- Before converting tasks, read `tools/perplexity` at the pinned commit for the exact input formats and
  scoring (e.g. HellaSwag length normalisation vs lm-eval `acc_norm`), and record any difference here.

### D26. Generation-sanity prompts — planned, open
- Proposal: 50 items drawn by seed from the selection questions of all five tasks, stored with the splits.

---

## Open

- Team value for R_max (`configs/budgets.yaml`), required before Phase C.
- Team confirmation of task accuracy as primary endpoint and the 0.5-point margin (D5).
- Whether to keep one Pi energy check (D2 removed energy; question raised, not answered).
- Whether to pin `torch==2.13.0` on every platform (D13).
- How to handle sensitivity precision for Phase C (D31: accept N = 48, enlarge the split, or a
  solver-based rule).
