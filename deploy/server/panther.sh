#!/usr/bin/env bash
# Run Panther on the server as svcuser, in rootless Podman. Makes no host changes: all images, containers,
# data and logs live under $PANTHER_WORK (default /s3/panther/work), which uninstall.sh removes.
#
#   panther.sh build                   build the toolchain image (skipped if this exact image exists)
#   panther.sh smoke                   check both container kinds: CPU (network, confined) and GPU
#   panther.sh run CMD...              CPU container, foreground, throwaway
#   panther.sh gpurun CMD...           GPU container, foreground, throwaway
#   panther.sh job NAME CMD...         CPU container, detached as panther-NAME (survives logout)
#   panther.sh gpujob NAME CMD...      GPU container, detached as panther-NAME
#   panther.sh logs NAME               follow a job's output
#   panther.sh stop NAME               stop a job
#   panther.sh status                  jobs, GPU, memory, GitLab health
#   panther.sh clean                   remove exited job containers and dangling images
#
# Two container kinds, so no container has both the GPU and a network:
#   CPU: full SELinux confinement, user-mode network (downloads). No GPU.
#   GPU: the NVIDIA devices, which SELinux only allows with labelling off (--security-opt label=disable),
#        so these get no network at all (--network none, HF offline). Download first in a CPU job.
# Both: rootless (keep-id), all capabilities dropped, no-new-privileges, read-only root filesystem,
# only repo (read-only) and data mounted, lower CPU/IO priority.
#
# Resource caps (env): PANTHER_CPUS (default 20 of 48 threads), PANTHER_MEM (default 64g).
# Don't run large jobs while the LLM stack (llm-llama, 88g cap) is up: together they would exceed RAM.
set -euo pipefail
# Site settings (host, user, health-check URL) live in the git-ignored deploy/server/local.env.
PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"

W="${PANTHER_WORK:-/s3/panther/work}"
REPO="$W/repo"
DATA="$W/data"
CPUS="${PANTHER_CPUS:-20}"
MEM="${PANTHER_MEM:-64g}"
IMAGE=localhost/panther
GITLAB="${PANTHER_HEALTH_URL:?set PANTHER_HEALTH_URL in deploy/server/local.env}"

[ -d "$W" ] && [ -w "$W" ] || { echo "$W missing or not writable; run host-setup.sh as root first" >&2; exit 1; }
mkdir -p "$DATA/logs" "$W/tmp" "$W/containers/storage"
export TMPDIR="$W/tmp"   # image layer copies would otherwise go to /var/tmp (small /var)

P=(podman --root "$W/containers/storage" --runroot "/run/user/$(id -u)/panther" --events-backend file
   --tmpdir "$W/tmp/libpod")

llama_commit() { awk '/^llama_cpp:/{f=1} f&&/commit:/{print $2; exit}' "$REPO/configs/models.yaml"; }

image_tag() {
  # Content hash of everything baked into the image.
  { cat "$REPO/deploy/server/Containerfile" "$REPO/pyproject.toml" "$REPO/uv.lock"; llama_commit; } \
    | sha256sum | cut -c1-12
}

gitlab_code() { curl -sL -o /dev/null -w '%{http_code}' --max-time 20 "$GITLAB" || true; }

run_args() {  # run_args cpu|gpu
  RUN_ARGS=(
    --userns keep-id
    --cap-drop all --security-opt no-new-privileges
    --read-only --tmpfs /tmp:rw,size=16g --shm-size 2g --pids-limit 4096
    --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM"
    -v "$REPO:/work/repo:ro,z" -v "$DATA:/work/data:rw,z"
    -e "PANTHER_THREADS=$CPUS"
    -e "PANTHER_IMAGE_ID=$("${P[@]}" image inspect -f '{{.Id}}' "$IMAGE:current" 2>/dev/null || echo unknown)"
  )
  [ -n "${PANTHER_FORMATS:-}" ] && RUN_ARGS+=(-e "PANTHER_FORMATS=$PANTHER_FORMATS")
  [ -n "${PANTHER_HF_REPO:-}" ] && RUN_ARGS+=(-e "PANTHER_HF_REPO=$PANTHER_HF_REPO")
  [ -n "${PANTHER_ANCHOR:-}" ] && RUN_ARGS+=(-e "PANTHER_ANCHOR=$PANTHER_ANCHOR")
  case "$1" in
    cpu) RUN_ARGS+=(--network pasta) ;;
    gpu) RUN_ARGS+=(--device nvidia.com/gpu=all --security-opt label=disable --network none
                    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1) ;;
    *) echo "run_args: $1" >&2; exit 2 ;;
  esac
}

start_job() {  # start_job cpu|gpu NAME CMD...
  local kind="$1" name="$2"; shift 2
  [ "$#" -gt 0 ] || { echo "usage: panther.sh ${kind/cpu/}job NAME CMD..." >&2; exit 2; }
  [ "$(gitlab_code)" = 200 ] || { echo "GitLab is not returning 200; not starting a job" >&2; exit 1; }
  run_args "$kind"
  # Output goes to the capped k8s-file log and, in full, to data/logs/NAME.log.
  "${P[@]}" run -d --name "panther-$name" --label "panther.kind=$kind" "${RUN_ARGS[@]}" \
    --log-driver k8s-file --log-opt max-size=20mb \
    "$IMAGE:current" bash -o pipefail -c '"$@" 2>&1 | tee -a "/work/data/logs/$0.log"' "$name" "$@"
  echo "started $kind job panther-$name; log: $DATA/logs/$name.log"
}

foreground() {  # foreground cpu|gpu CMD...
  local kind="$1"; shift
  run_args "$kind"
  local tty=(); [ -t 0 ] && tty=(-it)
  "${P[@]}" run --rm "${tty[@]}" "${RUN_ARGS[@]}" "$IMAGE:current" "$@"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  build)
    tag="$(image_tag)"
    if "${P[@]}" image exists "$IMAGE:$tag"; then
      echo "image $IMAGE:$tag already built"
    else
      "${P[@]}" build --cpu-period 100000 --cpu-quota $((CPUS * 100000)) --memory "$MEM" \
        --build-arg "LLAMA_COMMIT=$(llama_commit)" --build-arg "JOBS=$CPUS" \
        -t "$IMAGE:$tag" -f "$REPO/deploy/server/Containerfile" "$REPO"
    fi
    "${P[@]}" tag "$IMAGE:$tag" "$IMAGE:current"
    echo "current -> $IMAGE:$tag"
    ;;
  smoke)
    echo "== CPU container (confined, network, no GPU)"
    foreground cpu bash -c '
      set -e
      [ ! -e /dev/nvidia0 ] && echo "no GPU device: ok"
      curl -s -o /dev/null -w "huggingface.co %{http_code}\n" https://huggingface.co
      /opt/llama.cpp/build-cpu/bin/llama-cli --version 2>&1 | head -1
      $PANTHER_PY -c "import panther, gguf, ortools; print(\"python ok\")"
      touch /work/data/.smoke && rm /work/data/.smoke && echo "data writable"'
    echo "== GPU container (GPU, no network)"
    foreground gpu bash -c '
      set -e
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
      /opt/llama.cpp/build-cuda/bin/llama-cli --list-devices 2>&1 | grep -i cuda
      $PANTHER_PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"
      if curl -s -m 5 -o /dev/null https://huggingface.co; then echo "NETWORK REACHABLE: should not be" >&2; exit 1; else echo "no network: ok"; fi'
    ;;
  run)    foreground cpu "$@" ;;
  gpurun) foreground gpu "$@" ;;
  job)    start_job cpu "$@" ;;
  gpujob) start_job gpu "$@" ;;
  logs)  "${P[@]}" logs -f "panther-$1" ;;
  stop)  "${P[@]}" stop -t 30 "panther-$1" ;;
  status)
    "${P[@]}" ps -a --format '{{.Names}}\t{{.Labels}}\t{{.Status}}\t{{.RunningFor}}'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    free -g | awk 'NR<=2'
    uptime
    echo "gitlab: $(gitlab_code)"
    ;;
  clean)
    "${P[@]}" container prune -f
    "${P[@]}" image prune -f
    ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
