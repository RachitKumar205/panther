# Panther on the server

How Panther runs on the shared server (RHEL 10, SELinux enforcing, RTX PRO 6000 Blackwell 96 GB, EPYC 9224 24C/48T,
124 GB RAM), which also hosts production GitLab.

## Running contests

Everything goes through `./panther` at the repo root (its usage header lists the commands; `./panther` alone prints
it): from your machine it syncs the code here and runs the same command. `./panther run 14b@5` returns at once and
runs the contest detached as three containers in order: `fetch` (CPU, network), `run` (GPU, no network), `publish`
(CPU, network). A run claims its GPU and RAM when it starts
(`data/locks/claims.json`) and refuses to start, naming the runs holding the rest, if they do not fit beside the
active ones; `./panther stop 14b@5` stops only that run. Starting a run also starts the dashboard and the GitLab guard
(3 bad checks → every Panther container paused; 5 good → resumed). Log: `data/logs/panther.log` (also on the
dashboard); each step's log: `data/logs/<size>-b<budget>-<step>.log`.

## Site settings

Host, account and health-check URL are not in the repository. Copy `local.env.example` to `local.env`
(git-ignored) next to it and fill it in; `./panther` reads it on your machine, and
it is synced to the server, where `./panther`, `host-setup.sh` and `uninstall.sh` read it.
Below, `svcuser` stands for that account and "the server" for that host.

## Principles

- **Nothing installed on the host.** All tools run in one rootless Podman image.
- **Three host changes in total**, made once by an admin as root, each logged with its undo.
- **Removable.** `uninstall.sh` restores the host exactly, from the ledger.
- **GitLab first.** Containers are capped, deprioritised, wait to start while GitLab isn't healthy, and the guard in `./panther` pauses
  them if it degrades.

## Layout on the host

```
/s3/panther/                       root, 755
├── HOST_CHANGES.tsv               ledger: timestamp <TAB> undo command, one line per host change
└── work/                          svcuser, 750
    ├── repo/                      code, synced from your machine by ./panther (mounted read-only in containers)
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

No packages, firewall rules, Quadlets or system services; the only proxy change is the optional dashboard below. The script is idempotent, checks
GitLab returns 200 before and after, and undoes its own run on any failure. `--dry-run` prints the steps.

Applied 2026-11-10; ledger:
```
REMOVE_BASE
semanage fcontext -d -e /var/lib/containers /s3/panther/work/containers
loginctl disable-linger svcuser
```

## The dashboard

`./panther dashboard` (also started by the first `./panther run`) keeps `panther-dashboard` running: a small CPU container
(2 CPUs, 1 GB) serving `scripts/server/dashboard.py` with `data/` mounted read-only. Pause and stop leave it
running. It listens only at `PANTHER_DASHBOARD_BIND` on the server: loopback for an SSH tunnel
(`ssh -L 43800:localhost:43800 …`), or the gateway of the network the server's reverse proxy sits on, so the proxy
reaches it and the LAN does not. The page is read-only and public; the server's account and address and the shared
service's domain (from `local.env`) are masked in everything it sends, and at most 16 live log streams run at once.

A public URL goes through the server's existing reverse proxy (a root nginx container in front of GitLab, configs
in `/etc/nginx/conf.d`, Let's Encrypt certificates renewed by a certbot timer). `proxy-setup.sh`, run once as root
after the DNS record exists and the dashboard is running, makes three more host changes, each logged with its undo:

| # | Change | Undo (in ledger) |
|---|---|---|
| 4 | `/etc/nginx/conf.d/panther-http.conf` (ACME challenge, redirect to HTTPS), then `nginx -t` and a graceful reload | remove it, `nginx -t`, reload |
| 5 | a certificate named `panther` (the certbot image the renewal timer uses; renewed with the others) | `certbot delete --cert-name panther` |
| 6 | `/etc/nginx/conf.d/panther.conf` (HTTPS; GET only; per-address request and connection limits) | remove it, `nginx -t`, reload |

It refuses to start if GitLab isn't 200, the proxy's config doesn't pass `nginx -t`, the name doesn't resolve, the
proxy can't reach the dashboard, or either file exists; it checks the page and GitLab return 200 afterwards and
undoes its own run on any failure. `--dry-run` prints the steps. Templates: `nginx/`.

## The image (Containerfile)

- Base `nvidia/cuda:12.9.1-devel-ubi9`, uv `0.9.5`, both pinned by digest.
- llama.cpp at the commit in `configs/models.yaml`, shallow-fetched and checked: `build-cuda`
  (`sm_120`, used for every measurement) and `build-cpu` (`GGML_NATIVE`; nothing uses it any more, kept until the
  next deliberate rebuild).
- `/opt/venv`: Panther's dependencies from `uv.lock` (torch 2.13.0+cu129).
  `/opt/venv-convert`: llama.cpp's converter requirements at the same commit.
- Source code is **not** in the image; it is mounted from `work/repo`, so code changes need no rebuild.
- Tag = hash of Containerfile + pyproject.toml + uv.lock + llama.cpp commit; `current` points at the
  latest build (~17 GB).

## Two kinds of container

| | CPU (`fetch`, `publish`, `cexec`) | GPU (`run`, `test`, `exec`) |
|---|---|---|
| SELinux | confined (`container_t`) | labelling off (required for `/dev/nvidia*`) |
| GPU | none | all (CDI `nvidia.com/gpu=all`) |
| Network | rootless user-mode (pasta) | **none**; HF offline |
| Use | downloads, CPU-only processing | imatrix, quantization, evaluation |

Both: rootless with `--userns keep-id` (files owned by svcuser), all capabilities dropped,
`no-new-privileges`, read-only root filesystem, `/tmp` tmpfs 16 GB, pids ≤ 4096, ≤ `PANTHER_CPUS` CPUs (40), RAM
capped (a run's containers: its claim; no swap), `nice 10` + `ionice` best-effort 7. Only `work/repo` (read-only) and `work/data` are mounted.

No container ever has both the GPU and a network. Anything needing the network runs first as a CPU job.

## Commands

`./panther` alone prints the help. The SSH connection from your machine must be open (a control master, opened once
with your hardware key). Never poll the server in an unguarded loop or let SSH fall back to a password login: its
sshd blocks an address after repeated failed logins. CPU cap per container: `PANTHER_CPUS`.

## Removing everything

```bash
bash /s3/panther/work/repo/deploy/server/uninstall.sh --dry-run   # as root: show what it would do
bash /s3/panther/work/repo/deploy/server/uninstall.sh             # asks before deleting /s3/panther
```

It stops svcuser's host-side Panther loops (each run's `panther _run` and the guard `panther _guard`, matched by
user and exact command line, never by a broad pattern), removes Panther's containers and images (only from Panther's
storage), replays the ledger newest first (each undone step's line is dropped, so a rerun after a partial uninstall
picks up where it stopped), removes `/s3/panther` after confirmation, and checks the SELinux rule is gone and GitLab returns 200.
