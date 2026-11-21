# Running Panther (part 8)

One command on your machine runs a whole contest on the server (anton):

```bash
./panther build             # once, and after a dependency or llama.cpp change: the image, on the server
./panther run 14b@5         # a contest, detached: fetch, then run (the model's size @ the RAM budget in GiB)
./panther run 14b@5 --qat 1024   # the same, with a QAT finalist trained on 1,024 samples (part 6)
./panther logs 14b@5        # follow its log
./panther stop 14b@5        # stop it; ./panther run 14b@5 again continues where it stopped
./panther status            # Panther's containers, the GPU, the disk
./panther test              # the tests, on your machine
```

From your machine, `./panther` copies the code to the server (rsync over your SSH connection) and runs the same
command there. The run's progress, and its verdict line at the end, are in `data/logs/14b@5.log`; every result is in
`data/contests/<model>@<budget>/` (`preset.json`, `search.json`, `winner.json`, `verdict.json`, `winner.gguf`).

## The entrypoint

Inside the image, everything is one Python entrypoint (`src/panther/run.py`), which calls the modules in pipeline
order:

| Command | Container | What |
|---|---|---|
| `python -m panther fetch 14b` | **fetch**: network, no GPU | the dataset (part 0) and the model's weights (part 1's download) |
| `python -m panther run 14b@5 [--qat N]` | **run**: GPU, no network | the model's type copies (part 1) → the preset (part 2, on the winner set) → the search (part 4, on the GA set) → optionally QAT on the best finalist, then distilled (part 6) → the winner (part 7: the finalists built and distilled, on the winner set) → the verdict (part 7, on the verdict set) |

Every step skips what it has already made, so running a contest again continues where it stopped. A failure stops
the run with its traceback in the log. No container has both the network and the GPU.

The run refuses to start if torch sees no GPU (a driver too old for torch's CUDA build would otherwise let some
parts run, slowly, on the CPU).

## What you provide

`deploy/server/local.env` (git-ignored): copy `deploy/server/local.env.example` and fill it in. The variable names
are the earlier setup's, so an existing `local.env` keeps working.

Open the SSH connection once per session, as before, e.g.
`ssh -fNM -o ControlPath=$HOME/.ssh/cm-panther user@server`.

## The server (assumed set up, as it already is)

The earlier `host-setup.sh` has been run on anton (once, as root): `/s3/panther/work` owned by the Panther account,
container storage there with SELinux container labels, lingering enabled so runs survive the SSH session. Rootless
Podman with the NVIDIA container device (CDI). Nothing else is installed on the host.

```
/s3/panther/work/
├── repo/                 the code, synced by ./panther (mounted read-only into containers)
├── data/                 everything produced (mounted read-write): models/, contests/, dataset, logs/, scratch/
├── containers/storage/   Panther's Podman storage
└── tmp/                  build temp
```

## The image (`deploy/server/Containerfile`)

- CUDA 12.9 (UBI 9) base, pinned by digest; Python 3.12; uv.
- **llama.cpp at the commit pinned in `configs/models.yaml`**, CUDA build for the RTX PRO 6000 (sm_120): the programs
  Panther calls by name (`llama-quantize`, `llama-imatrix`, `llama-perplexity`, `llama-server`, `llama-tokenize`) on
  `PATH`, and `libggml-base.so` (whose quantizer GPTQ calls) on the library path.
- **The converter** (`convert_hf_to_gguf.py`) runs in its own environment from llama.cpp's requirements at the same
  commit: they pin an older transformers than Panther's. `PANTHER_CONVERT` points at `deploy/server/convert.py`, which
  runs the converter under that environment's Python.
- **Panther's environment: exactly `uv.lock`** (`uv sync --frozen`). The code itself is not in the image: it is
  mounted at `/work/repo`, so a code change needs no rebuild.
- `PANTHER_DATA=/work/data`, `HF_HOME` under it. Lower CPU and I/O priority than everything else on the server.

## The containers

Rootless Podman, as before: the user namespace kept, every capability dropped, no new privileges, a read-only root,
the code read-only, the data read-write. `--cpus 40` (of 48 threads), memory `PANTHER_RAM` (default 100 GB) for the
run, 32 GB for fetch. The run container gets the GPU (`--device nvidia.com/gpu=all`), no network, and
`HF_HUB_OFFLINE=1`.

**Candidate files** (the search builds 8 at a time, each the size of the budget) go to `data/scratch/`
(`PANTHER_SCRATCH`), not `/dev/shm`. A file just written stays in the page cache, so loading it is a memory read
anyway, and the page cache gives way under memory pressure where `/dev/shm` would not: the 24B @ 7 search's 8 files
are ~56 GB.

## Kept from the earlier setup, and dropped

| Kept | Dropped, and why |
|---|---|
| `./panther` syncs and runs the same command on the server; `local.env` and its names | **Claims** of GPU and RAM across runs: one contest at a time; inside a run, `Resources` shares GPU and RAM between its own jobs |
| The image: the pinned CUDA base, llama.cpp at the pinned commit (CUDA build), the converter's own environment, the lock file | **The CPU build of llama.cpp**: nothing uses it |
| Rootless Podman, the same security options and mounts, `nice`/`ionice` | **The GitLab guard** (pause on an unhealthy GitLab): not part of our pipeline; the lower priority and the CPU and memory caps keep GitLab's share |
| fetch (network) and run (GPU) as separate containers | **The publish step** (Hugging Face upload, store backup) and **the Pi** check: not built yet (decision D: no store backup) |
| Detached runs, one log per contest, stop and continue | **The dashboard**, `pause`/`resume`, `exec`: not needed to run the pipeline |

## Before the first run on anton: check

- **torch's CUDA build.** `uv.lock` has torch 2.15 built for CUDA 13.2, which needs a recent NVIDIA driver; the
  earlier setup pinned a CUDA 12.9 build of torch for this server. `./panther run` stops at once if torch sees no GPU.
- **Memory.** `Resources` reads the free memory of the whole server (psutil), not the container's limit, so with
  `PANTHER_RAM=100g` it can plan more than the container may use.

## Tests (`tests/run/`)

Every module replaced by a recording stand-in (no network, no GPU): `fetch` builds the dataset and downloads the model;
`run` calls the parts in order with the right sets (the distill set for the model, the winner set for the preset and
the winner, the GA set for the search, the verdict set for the verdict); QAT trains the best finalist, distills it and
adds it to the winner's files, and is not redone; a badly written contest fails; a run without a GPU fails before doing
anything.
