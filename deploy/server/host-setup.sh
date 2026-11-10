#!/usr/bin/env bash
# Root-only host changes Panther needs on the server. Run as root:
#
#   bash /path/to/host-setup.sh [--dry-run]
#
# What it changes (and nothing else: no packages, firewall, nginx, Quadlets or system services):
#   1. Creates /s3/panther (root-owned) with /s3/panther/work owned by svcuser, for repo, data and
#      rootless container storage. /var is too small and svcuser can't write to /s3.
#   2. Adds an SELinux file-context equivalence so rootless container storage under
#      /s3/panther/work/containers gets the same labels as /var/lib/containers.
#   3. Enables systemd linger for svcuser, so long jobs survive the SSH session ending
#      (only if it is not already on).
#
# Idempotent. Each change is appended to /s3/panther/HOST_CHANGES.tsv with the command that undoes it,
# as it is made; uninstall.sh replays that ledger in reverse. If anything fails, or GitLab stops
# returning 200, this run's changes are undone.
set -Eeuo pipefail

USER_NAME="${PANTHER_USER:-svcuser}"
BASE="${PANTHER_BASE:-/s3/panther}"
W="$BASE/work"
LEDGER="$BASE/HOST_CHANGES.tsv"
GITLAB=https://gitlab.example.com
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

die() { echo "host-setup: $*" >&2; exit 1; }
do_() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
gitlab_code() { curl -sL -o /dev/null -w '%{http_code}' --max-time 20 "$GITLAB" || true; }

RUN_UNDO=()
record() {  # record UNDO-COMMAND: log a change made in this run
  RUN_UNDO+=("$1")
  local now; now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [ "$DRY" = 1 ] || printf '%s\t%s\n' "$now" "$1" >> "$LEDGER"
}

rollback() {
  local rc=$?
  # set -E makes subshells inherit this trap; only the main shell may undo anything.
  [ "$BASHPID" = "$$" ] || exit "$rc"
  trap - ERR
  echo "host-setup: failed (exit $rc); undoing this run's changes" >&2
  local i
  for ((i = ${#RUN_UNDO[@]} - 1; i >= 0; i--)); do
    local u="${RUN_UNDO[$i]}"
    if [ "$u" = "REMOVE_BASE" ]; then
      echo "+ rm -rf --one-file-system $BASE" >&2; [ "$DRY" = 1 ] || rm -rf --one-file-system "$BASE"
    else
      echo "+ $u" >&2; [ "$DRY" = 1 ] || bash -c "$u" || echo "  undo failed: $u" >&2
      # Drop the line from the ledger once undone.
      [ "$DRY" = 1 ] || { [ -f "$LEDGER" ] && grep -vF -- "	$u" "$LEDGER" > "$LEDGER.tmp" && mv "$LEDGER.tmp" "$LEDGER" || true; }
    fi
  done
  exit "$rc"
}
trap rollback ERR

# --- preconditions (read-only) ---
[ "$(id -u)" = 0 ] || die "run as root"
mountpoint -q /s3 || die "/s3 is not mounted (unlock it first)"
id "$USER_NAME" >/dev/null || die "no user $USER_NAME"
command -v semanage >/dev/null || die "semanage missing"
[ "$(gitlab_code)" = 200 ] || die "GitLab is not returning 200 before any change; not touching anything"
if [ -e "$BASE" ] && [ ! -f "$LEDGER" ]; then
  die "$BASE exists but has no ledger: not created by this script, refusing to adopt it"
fi

# --- 1. project folder ---
if [ ! -d "$BASE" ]; then
  do_ mkdir -m 755 "$BASE"
  record "REMOVE_BASE"
fi
if [ ! -d "$W" ]; then
  do_ mkdir -m 750 "$W"
  do_ chown "$USER_NAME:$USER_NAME" "$W"
fi
for d in repo data containers tmp; do
  [ -d "$W/$d" ] || { do_ mkdir -m 750 "$W/$d"; do_ chown "$USER_NAME:$USER_NAME" "$W/$d"; }
done

# --- 2. SELinux labels for rootless container storage on /s3 ---
if ! semanage fcontext -l -C | grep -qF "$W/containers = /var/lib/containers"; then
  do_ semanage fcontext -a -e /var/lib/containers "$W/containers"
  record "semanage fcontext -d -e /var/lib/containers $W/containers"
fi
do_ restorecon -RF "$W/containers"

# --- 3. linger ---
if [ "$(loginctl show-user "$USER_NAME" -p Linger --value 2>/dev/null || echo no)" != yes ] \
   && [ ! -e "/var/lib/systemd/linger/$USER_NAME" ]; then
  record "loginctl disable-linger $USER_NAME"   # before: a half-applied enable still gets undone
  do_ loginctl enable-linger "$USER_NAME"
fi

# --- post-check ---
[ "$(gitlab_code)" = 200 ] || { echo "host-setup: GitLab not 200 after changes" >&2; false; }

trap - ERR
echo "host-setup: done. Ledger: $LEDGER"
[ "$DRY" = 1 ] || cat "$LEDGER"
