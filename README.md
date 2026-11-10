# Panther

Find the best model, and its best mixed-precision quantization, that fits a RAM budget on an 8 GB
Raspberry Pi 5. See [PLAN.md](PLAN.md).

## Layout

| Path | What |
|---|---|
| `configs/` | Pinned model revisions and llama.cpp commit, format menu, budgets |
| `src/panther/models` | Model geometry, tensor/gene inventory, divisibility checks |
| `src/panther/costs` | Exact format byte sizes (from gguf-py) and the RAM price-tag model |
| `src/panther/gguftools` | Anchored `--tensor-type` files, mixed-GGUF assembly from a per-format tensor store, type read-back |
| `src/panther/search` | Allocation problem, CP-SAT solver with top-k shortlist, greedy and random baselines |
| `src/panther/stats` | Pre-registered decision rule (paired bootstrap, Holm) |
| `scripts/ram_fit_table.py` | Writes `docs/ram_fit_table.md` |
| `scripts/server/` | Server jobs (run inside the container), e.g. tensor-store build |
| `deploy/server/` | Container image, rootless run wrapper, root host-setup and uninstall for the server ([details](deploy/server/README.md)) |
| `docs/PIPELINE.md` | How the whole method works, stage by stage, with current status and numbers |
| `docs/DECISIONS.md` | Every decision with alternatives and reasons; steps are logged here before they run |

## Local (dev machine)

```bash
uv sync
uv run pytest
uv run python scripts/ram_fit_table.py
```

## Server (the server)

Everything runs in one rootless Podman image (CUDA 12.9 UBI9 base, llama.cpp CPU + CUDA builds at the
pinned commit, Python envs from `uv.lock`). Nothing is installed on the host. All of Panther's files,
images and containers live under `/s3/panther`.

```bash
# once, as root: creates /s3/panther, SELinux label equivalence for container storage, linger for svcuser.
# Every change is logged with its undo in /s3/panther/HOST_CHANGES.tsv.
bash host-setup.sh

deploy/server/push.sh                                     # from the Mac: rsync code to /s3/panther/work/repo
/s3/panther/work/repo/deploy/server/panther.sh build      # on the server, as svcuser
/s3/panther/work/repo/deploy/server/panther.sh smoke
/s3/panther/work/repo/deploy/server/panther.sh job fetch-1.5b scripts/server/fetch_model.sh Qwen2.5-1.5B-Instruct
/s3/panther/work/repo/deploy/server/panther.sh gpujob store-1.5b \
  scripts/server/build_tensor_store.sh Qwen2.5-1.5B-Instruct /work/data/splits/calibration.txt

# remove everything, as root (asks before deleting /s3/panther)
bash /s3/panther/work/repo/deploy/server/uninstall.sh
```

GPU jobs (`gpujob`, `gpurun`) get the NVIDIA devices with SELinux labelling off and no network; CPU jobs
(`job`, `run`) are fully confined and have the network, so downloads happen there first.
Jobs run at lower CPU/IO priority, capped at 20 CPUs and 64 GB (`PANTHER_CPUS`, `PANTHER_MEM`), and
refuse to start unless GitLab returns 200.

Not built yet: data splits, quality pipeline (KL / perplexity / tasks), sensitivity, Pi driver and agent.
