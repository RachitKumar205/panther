# Decisions that stand

The full history (D1–D51, with alternatives and reasons) is in git, up to commit `756ef18`. This page keeps only
what is true now.

## Scope
- **Goal:** for each RAM budget (1, 3, 5, 7 GiB) on a Raspberry Pi 5 (8 GB), the best Qwen2.5 Instruct model
  (1.5B, 3B, 7B, 14B, 32B) with one llama.cpp quantization type per weight matrix. Energy, time to first token and
  load time are out of scope.
- **Tools:** llama.cpp pinned at `cd959bae` (v0.9.0); models pinned by Hugging Face revision (`configs/models.yaml`).
- **Machines:** quantization and quality tests on the server (anton, RTX PRO 6000); the Pi only measures RAM and
  speed. The Mac relays files between them (they cannot reach each other).

## Server rules (user)
- Everything in rootless Podman; nothing installed on the host; every host change logged with its undo and
  reversed by `deploy/server/uninstall.sh`. Ask before changing anything on the host.
- GitLab runs on the same machine and must keep returning 200: jobs check it before starting and `guard.sh`
  pauses Panther's containers if it degrades. Never `firewall-cmd --reload`.
- CPU jobs are confined and have the network (downloads); GPU jobs run with SELinux labelling off and no network.
- Never retry SSH in a loop (the server locks out repeated failures); wait with `wait-job.sh`.

## Method (docs/STANDARD.md)
- **RAM model:** weights + KV cache (q8_0, 4,096 tokens) + runtime overhead (0.40 GiB until the Pi measures it).
- **Solution space:** only genotypes within −1% / +0.5% of the budget.
- **Quantization types per model and budget:** from 1.5 bits/weight below to 2.5 above the budget's average.
  Types that need importance-matrix data are not used without it (token embedding; the output layer unless the
  importance matrix covers it, which only 32B's does).
- **Part costs:** KL of each part (attention or feed-forward of a layer, embedding, output layer) moved from the
  budget's starting type to a higher type, then 2 rounds measuring the types the solver chooses (±1). Choosing
  from the weights alone reaches ~75% of the achievable improvement on 14B (about llama.cpp-recipe level); this
  reaches ~97%.
- **List:** the solver's 6 best genotypes per model (≥5% apart) + one padded-uniform baseline; stored once.
- **Tests:** each genotype built once; KL (selection text), MMLU 1,000, HellaSwag 2,000, ARC-C 1,172,
  WinoGrande 1,267, GSM8K 300 (fixed test subset, approved by the user), perplexity; all on the GPU.
- **Winner:** best mean accuracy; genotypes the paired bootstrap cannot separate from it (95% interval of the lead
  reaching below 0.5 points) are tied, and the lowest KL among them wins (2026-11-11: 14B's padded uniform 'won'
  by 0.08 points with nearly twice the KL). Models compared by paired bootstrap on accuracy.
- **Baseline:** for every model and budget, the best existing quantization that fits: the better of llama.cpp's two
  largest fitting presets, same importance matrix (2026-11-12).
- **Pi:** winners reach it through a private Hugging Face repo per model and budget (the VPN to the server ran at
  0.5 MB/s). It loads the winner once, peak RAM against the budget, speed (256-token prompt, 64 out, 4 threads).

## Pipeline
- One command per budget (`deploy/server/run-budget.sh B`); one lane per model, straight through to its winner
  and Pi bundle; budgets can run at the same time (per-model locks, per-budget job names); every step skips work
  already on disk; heavy steps wait for 30 GB free disk.
- Part-cost files are reflink copies of the starting-type file with the changed part appended, used only after a
  tensor-by-tensor match with a normally built file.

## Models (2026-11-12)
- Qwen2.5 1.5B–32B; plus InternLM2.5 20B (7 and 5 GiB) and Mistral Small 24B (7 GiB) to place ~20B models.
  No ~20B model fits 3 GiB (InternLM 20B needs ≥ 4.6 GiB at the lowest types); InternLM2.5 7B stands in.
  Reka Flash 3.1 dropped: its feed-forward width is not a multiple of 256.

## Search start (2026-11-12)
- Below ~2.5 bits per weight the uniform-start search loses to llama.cpp's presets (3 GiB 7B, 7 GiB InternLM 20B)
  or to its own uniform baseline (1 GiB, 7 GiB Mistral 24B); causes in RESULTS.md.
- `PANTHER_ANCHOR=preset`: part costs are measured from the budget's best fitting preset (its per-matrix types
  read from the preset file, mapped to the nearest menu type), one type up and one down per block, then the usual
  two rounds; the preset's layout, padded into the RAM window, is listed as a candidate. Files get `-anchored`; the
  uniform-start results stay. Run first on the ~2-bit contests (InternLM 20B and Mistral 24B at 7 GiB, then
  InternLM 20B at 5 GiB and Qwen 7B at 3 GiB).
- Perplexity is compared only between models of one family (one tokenizer); across families the pick goes by
  accuracy, then RAM.

## Results so far
- Full results: [RESULTS.md](RESULTS.md).
- Proof of concept (7 GiB, f16 KV): 14B mixed (75.2% mean accuracy) beat 7B (70.4%) and uniform 14B (73.5%), and
  llama.cpp's IQ3_XXS recipe for 14B (74.1%, 6.68 GiB): +1.05 points, CI [0.00, 2.10], lower KL and perplexity.
