# Panther on the server

How Panther runs on the shared server (RHEL 10, SELinux enforcing, RTX PRO 6000 Blackwell 96 GB, EPYC 9224 24C/48T,
124 GB RAM), which also hosts production GitLab.

## Running a budget

```bash
deploy/server/run-budget.sh 7                                            # every searched model of the 7 GiB budget
deploy/server/run-budget.sh 7 Qwen2.5-14B-Instruct Mistral-24B-Instruct    # only these
```

One lane per model (`chain.sh`: one step after another, each a container job; a step whose job already exited 0 is
skipped and a running one adopted, so a lane can be restarted at any time). The steps are in
[scripts/server](../../scripts/server/README.md). Progress: `data/logs/progress-b<B>.log` and the status page.

## Site settings

Host, account and health-check URL are not in the repository. Copy `local.env.example` to `local.env`
(git-ignored) next to it and fill it in; `push.sh` reads it on your machine, and
`push.sh` copies it to the server, where `panther.sh`, `host-setup.sh` and `uninstall.sh` read it.
Below, `svcuser` stands for that account and "the server" for that host.

## Principles

- **Nothing installed on the host.** All tools run in one rootless Podman image.
- **Three host changes in total**, made once by an admin as root, each logged with its undo.
- **Removable.** `uninstall.sh` restores the host exactly, from the ledger.
- **GitLab first.** Jobs are capped, deprioritised, refuse to start if GitLab isn't healthy, and `guard.sh` pauses
  them if it degrades.

## Layout on the host

```
/s3/panther/                       root, 755
├── HOST_CHANGES.tsv               ledger: timestamp <TAB> undo command, one line per host change
└── work/                          svcuser, 750
    ├── repo/                      code, synced from the Mac by push.sh (mounted read-only in containers)
    ├── data/                      everything produced (mounted read-write): hf/, splits/, store/, logs/, cache/
    ├── containers/storage/        Panther's own Podman image/container storage (SELinux: container_var_lib_t)
    └── tmp/                       build temp (keeps image layers off /var)
```

## Host changes (host-setup.sh, run once as root)

| # | Change | Why | Undo (in ledger) |
|---|---|---|---|
| 1 | Create `/s3/panther`, `/s3/panther/work` owned by svcuser | `/var` too small; svcuser can't write `/s3` | remove `/s3/panther` (asks first) |
| 2 | `semanage fcontext -a -e /var/lib/containers /s3/panther/work/containers` | container storage on `/s3` needs container labels | `semanage fcontext -d -e …` |
| 3 | `loginctl enable-linger svcuser` (only if off) | jobs survive the SSH session ending | `loginctl disable-linger svcuser` |

No packages, firewall rules, nginx config, Quadlets or system services. The script is idempotent, checks
GitLab returns 200 before and after, and undoes its own run on any failure. `--dry-run` prints the steps.

Applied 2026-11-10; ledger:
```
REMOVE_BASE
semanage fcontext -d -e /var/lib/containers /s3/panther/work/containers
loginctl disable-linger svcuser
```

## The image (Containerfile)

- Base `nvidia/cuda:12.9.1-devel-ubi9`, uv `0.9.5`, both pinned by digest.
- llama.cpp at the commit in `configs/models.yaml`, shallow-fetched and checked: `build-cpu`
  (`GGML_NATIVE`, for final numbers) and `build-cuda` (`sm_120`, for imatrix and screening).
- `/opt/venv`: Panther's dependencies from `uv.lock` (torch 2.13.0+cu129).
  `/opt/venv-convert`: llama.cpp's converter requirements at the same commit.
- Source code is **not** in the image; it is mounted from `work/repo`, so code changes need no rebuild.
- Tag = hash of Containerfile + pyproject.toml + uv.lock + llama.cpp commit; `current` points at the
  latest build. Current: `localhost/panther:5b6c7bcf98f4` (~17 GB).

## Two kinds of container

| | CPU (`run`, `job`) | GPU (`gpurun`, `gpujob`) |
|---|---|---|
| SELinux | confined (`container_t`) | labelling off (required for `/dev/nvidia*`) |
| GPU | none | all (CDI `nvidia.com/gpu=all`) |
| Network | rootless user-mode (pasta) | **none**; HF offline |
| Use | downloads, CPU-only processing | imatrix, quantization, evaluation |

Both: rootless with `--userns keep-id` (files owned by svcuser), all capabilities dropped,
`no-new-privileges`, read-only root filesystem, `/tmp` tmpfs 16 GB, pids ≤ 4096, ≤ 20 CPUs, ≤ 64 GB RAM
(no swap), `nice 10` + `ionice` best-effort 7. Only `work/repo` (read-only) and `work/data` are mounted.

No container ever has both the GPU and a network. Anything needing the network runs first as a CPU job.

## Commands (as svcuser on the server)

```bash
cd /s3/panther/work/repo
deploy/server/panther.sh build        # build the image (no-op if this exact image exists)
deploy/server/panther.sh smoke        # checks both container kinds
deploy/server/panther.sh job NAME CMD...      # CPU job, detached; log in work/data/logs/NAME.log
deploy/server/panther.sh gpujob NAME CMD...   # GPU job, detached
deploy/server/panther.sh status       # jobs, GPU, memory, GitLab
deploy/server/panther.sh logs NAME | stop NAME | clean
```

From the Mac: `deploy/server/push.sh` syncs code (not data) to `work/repo` over the open SSH control master. Never
poll the server in an unguarded loop or let SSH fall back to a password login: its sshd blocks an address after
repeated failed logins, which also locks out the hardware key login.

Resource caps: `PANTHER_CPUS`, `PANTHER_MEM`. Don't run large jobs while the LLM stack (`llm-llama`,
88 GB) is running; together they exceed RAM.

## Removing everything

```bash
bash /s3/panther/work/repo/deploy/server/uninstall.sh --dry-run   # as root: show what it would do
bash /s3/panther/work/repo/deploy/server/uninstall.sh             # asks before deleting /s3/panther
```

It removes Panther's containers and images (only from Panther's storage), replays the ledger newest
first, removes `/s3/panther` after confirmation, and checks the SELinux rule is gone and GitLab returns 200.

## Smoke test (2026-11-10)

```
CPU: no GPU device, huggingface.co 200, llama.cpp cd959ba, python imports ok, data writable
GPU: RTX PRO 6000 Blackwell 97887 MiB, llama.cpp CUDA0 visible,
     torch 2.13.0+cu129 cuda 12.9 available, no network
```
