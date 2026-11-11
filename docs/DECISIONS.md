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
- **Winner:** best mean accuracy (ties: lower KL); models compared by paired bootstrap on accuracy.
- **Pi:** load the winner once, peak RAM against the budget, speed (256-token prompt, 64 out, 4 threads).

## Pipeline
- One command per budget (`deploy/server/run-budget.sh B`); one lane per model, straight through to its winner
  and Pi bundle; budgets can run at the same time (per-model locks, per-budget job names); every step skips work
  already on disk; heavy steps wait for 30 GB free disk.
- Part-cost files are reflink copies of the starting-type file with the changed part appended, used only after a
  tensor-by-tensor match with a normally built file.

## Results so far
- Proof of concept (7 GiB, f16 KV): 14B mixed (75.2% mean accuracy) beat 7B (70.4%) and uniform 14B (73.5%), and
  llama.cpp's IQ3_XXS recipe for 14B (74.1%, 6.68 GiB): +1.05 points, CI [0.00, 2.10], lower KL and perplexity.
