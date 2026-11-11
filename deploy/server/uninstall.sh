#!/usr/bin/env bash
# Remove Panther from the server completely and put the host back as it was. Run as root:
#
#   bash /s3/panther/work/repo/deploy/server/uninstall.sh [--dry-run] [--yes]
#
#   1. Stops and removes svcuser's Panther containers and images (they live only in Panther's own
#      container storage under /s3/panther/work/containers; svcuser's other Podman storage is untouched).
#   2. Replays /s3/panther/HOST_CHANGES.tsv in reverse: SELinux equivalence, linger, and finally the
#      /s3/panther folder itself (all data, models and results; asks first unless --yes).
#   3. Checks GitLab still returns 200.
#
# The script copies itself to a temporary file first, since it deletes the folder it lives in.
set -Euo pipefail
# Site settings (host, user, health-check URL) live in the git-ignored deploy/server/local.env.
export PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"

USER_NAME="${PANTHER_USER:?set PANTHER_USER in deploy/server/local.env}"
BASE="${PANTHER_BASE:-/s3/panther}"
W="$BASE/work"
LEDGER="$BASE/HOST_CHANGES.tsv"
GITLAB="${PANTHER_HEALTH_URL:?set PANTHER_HEALTH_URL in deploy/server/local.env}"
DRY=0; YES=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; --yes) YES=1 ;; esac; done

if [ -z "${PANTHER_UNINSTALL_COPY:-}" ]; then
  tmp="$(mktemp)"; cp "$0" "$tmp"
  PANTHER_UNINSTALL_COPY=1 exec bash "$tmp" "$@"
fi

trap 'rm -f "$0"' EXIT   # the temporary copy
die() { echo "uninstall: $*" >&2; exit 1; }
do_() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

[ "$(id -u)" = 0 ] || die "run as root"
[ -f "$LEDGER" ] || die "no ledger at $LEDGER: nothing recorded to undo"

# --- 1. containers and images (as the owning user, in Panther's storage only) ---
uid="$(id -u "$USER_NAME")"
if [ -d "/run/user/$uid" ] && [ -d "$W/containers/storage" ]; then
  P="podman --root $W/containers/storage --runroot /run/user/$uid/panther --events-backend file --tmpdir $W/tmp/libpod"
  do_ runuser -u "$USER_NAME" -- env "XDG_RUNTIME_DIR=/run/user/$uid" $P rm -af -t 30 || true
  do_ runuser -u "$USER_NAME" -- env "XDG_RUNTIME_DIR=/run/user/$uid" $P rmi -af || true
  do_ rm -rf "/run/user/$uid/panther"
fi
if findmnt -rn -o TARGET | grep -qF "$W/"; then
  die "something is still mounted under $W; stop it and rerun"
fi

# --- 2. ledger, newest first ---
mapfile -t UNDO < <(cut -f2- "$LEDGER" | tac)
fail=0
for u in "${UNDO[@]}"; do
  if [ "$u" = "REMOVE_BASE" ]; then
    du -sh "$BASE" 2>/dev/null || true
    if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then
      read -r -p "Permanently delete $BASE (all Panther data, models, results)? Type 'delete': " ans
      [ "$ans" = delete ] || { echo "kept $BASE; rerun to finish"; exit 1; }
    fi
    do_ rm -rf --one-file-system "$BASE" || { echo "  could not remove $BASE" >&2; fail=1; }
  else
    echo "+ $u"
    if [ "$DRY" != 1 ]; then
      bash -c "$u" || { echo "  undo failed: $u" >&2; fail=1; }
    fi
  fi
done

# --- 3. checks ---
[ "$DRY" = 1 ] || [ ! -e "$BASE" ] || { echo "$BASE still exists" >&2; fail=1; }
semanage fcontext -l -C 2>/dev/null | grep -qF "$W/containers" && { echo "SELinux equivalence still present" >&2; fail=1; }
code="$(curl -sL -o /dev/null -w '%{http_code}' --max-time 20 "$GITLAB" || true)"
echo "gitlab: $code"; [ "$code" = 200 ] || fail=1
[ "$fail" = 0 ] && echo "uninstall: host restored" || { echo "uninstall: finished with problems (above)" >&2; exit 1; }
