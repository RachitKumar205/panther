# Operations

> Target operations. Implemented in M6 (runtime/CLI) and M9 (deploy).

## Container topology

```
./panther run 14b@5
  └─ start: claim + detach _run (holds run-<id>.lock)
       ├─ fetch    CPU, network
       ├─ run      GPU, no network
       └─ publish  CPU, network
```

- Rootless podman, `--userns keep-id`, all capabilities dropped, `no-new-privileges`,
  read-only rootfs, only `repo` (RO) and `data` (RW) mounted.
- No container has both GPU and network.

## Claims

- A run reserves GPU and RAM for its whole life in `data/locks/claims.json`.
- A claim counts while its `run-<id>.lock` is held; the kernel releases it when the holder ends.
- Caps and the per-run reserve are a single config value (fixes B9/B14/B16).
- A refused claim is a "refused" log line, not a FAILED step.

## Guard

- Polls the shared service; 3 bad checks → `PAUSED.guard` and pause all Panther containers;
  5 good checks → resume.
- Manual `pause`/`resume` use `PAUSED` and never touch the dashboard.

## Deploy

- `deploy/server/Containerfile` at the pinned llama.cpp commit.
- `host-setup.sh`, `proxy-setup.sh`, `uninstall.sh` with a host-changes ledger and rollback.
- nginx vhosts for the optional public dashboard.
