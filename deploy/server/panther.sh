#!/usr/bin/env bash
# Panther on the server as svcuser, in rootless Podman; no host changes (everything lives under $PANTHER_WORK,
# default /s3/panther/work, which uninstall.sh removes). Two container kinds, so none has both GPU and network:
# CPU (SELinux-confined, network, no GPU) and GPU (no network). Both rootless, all capabilities dropped,
# read-only root, only the repo (read-only) and data mounted. Caps: PANTHER_CPUS (20), PANTHER_MEM (64g).
#
#   panther.sh build | status | clean          image (if changed) | jobs, GPU, GitLab | prune exited containers
#   panther.sh run|gpurun CMD...               foreground, throwaway
#   panther.sh job|gpujob NAME CMD...          detached as panther-NAME (log: data/logs/NAME.log)
#   panther.sh logs|stop|wait NAME             follow / stop / wait for a job (wait prints its exit code, removes it)
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
  [ -n "${PANTHER_HF_REPO:-}" ] && RUN_ARGS+=(-e "PANTHER_HF_REPO=$PANTHER_HF_REPO")
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
  run)    foreground cpu "$@" ;;
  gpurun) foreground gpu "$@" ;;
  job)    start_job cpu "$@" ;;
  gpujob) start_job gpu "$@" ;;
  logs)  "${P[@]}" logs -f "panther-$1" ;;
  stop)  "${P[@]}" stop -t 30 "panther-$1" ;;
  wait)  "${P[@]}" wait "panther-$1"; "${P[@]}" rm "panther-$1" >/dev/null ;;
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
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
