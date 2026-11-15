#!/usr/bin/env bash
# Puts Panther's dashboard behind the server's existing reverse proxy (the root nginx container that serves
# GitLab). Run as root, after `./panther dashboard` is running and the domain's DNS points at the server:
#
#   bash /path/to/proxy-setup.sh [--dry-run]
#
# What it changes (nothing else: no packages, firewall, Quadlets, nginx.conf or other sites):
#   1. /etc/nginx/conf.d/panther-http.conf: port 80 for the domain (ACME challenge, else redirect to HTTPS).
#   2. A Let's Encrypt certificate named "panther", issued with the certbot image the renewal timer already uses
#      (that timer renews it with the others).
#   3. /etc/nginx/conf.d/panther.conf: HTTPS for the domain, proxied to the dashboard.
# After each config change: `nginx -t` in the proxy, then a graceful reload (HUP, as the renewal timer does).
#
# Each change is appended to /s3/panther/HOST_CHANGES.tsv with its undo, as it is made; uninstall.sh replays that
# ledger in reverse. If anything fails, or GitLab stops returning 200, this run's changes are undone.
set -Eeuo pipefail
PANTHER_LOCAL_ENV="${PANTHER_LOCAL_ENV:-$(cd "$(dirname "$0")" && pwd)/local.env}"
# shellcheck source=/dev/null
[ -f "$PANTHER_LOCAL_ENV" ] && . "$PANTHER_LOCAL_ENV"

HERE="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="${PANTHER_DASHBOARD_DOMAIN:?set PANTHER_DASHBOARD_DOMAIN in deploy/server/local.env}"
UPSTREAM="${PANTHER_DASHBOARD_BIND:?set PANTHER_DASHBOARD_BIND in deploy/server/local.env}:43800"
GITLAB="${PANTHER_HEALTH_URL:?set PANTHER_HEALTH_URL in deploy/server/local.env}"
LEDGER="${PANTHER_BASE:-/s3/panther}/HOST_CHANGES.tsv"
CONF=/etc/nginx/conf.d
CERTS=/var/nginx/certs
WEBROOT=/var/nginx/webroot
CERTBOT=(podman run --rm -v "$CERTS:/etc/letsencrypt:z" -v "$WEBROOT:/var/www/certbot:z" docker.io/certbot/certbot)
RELOAD="podman exec nginx nginx -t -q && podman kill --signal HUP nginx >/dev/null"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

die() { echo "proxy-setup: $*" >&2; exit 1; }
do_() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
sh_() { echo "+ $1"; [ "$DRY" = 1 ] || bash -c "$1"; }
code() { curl -sL -o /dev/null -w '%{http_code}' --max-time 20 "$1" || true; }

RUN_UNDO=()
record() {  # record UNDO-COMMAND: log a change made in this run
  RUN_UNDO+=("$1")
  [ "$DRY" = 1 ] || printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$LEDGER"
}
rollback() {
  local rc=$?
  [ "$BASHPID" = "$$" ] || exit "$rc"
  trap - ERR
  echo "proxy-setup: failed (exit $rc); undoing this run's changes" >&2
  local i
  for ((i = ${#RUN_UNDO[@]} - 1; i >= 0; i--)); do
    local u="${RUN_UNDO[$i]}"
    echo "+ $u" >&2
    [ "$DRY" = 1 ] || bash -c "$u" || echo "  undo failed: $u" >&2
    [ "$DRY" = 1 ] || { grep -vF -- "	$u" "$LEDGER" > "$LEDGER.tmp" && mv "$LEDGER.tmp" "$LEDGER" || true; }
  done
  exit "$rc"
}
trap rollback ERR

render() {  # render TEMPLATE DEST
  sed -e "s|@DOMAIN@|$DOMAIN|g" -e "s|@UPSTREAM@|$UPSTREAM|g" "$HERE/nginx/$1" > "$2.tmp" && mv "$2.tmp" "$2"
}

# --- preconditions (read-only) ---
[ "$(id -u)" = 0 ] || die "run as root"
[ -f "$LEDGER" ] || die "no ledger at $LEDGER (run host-setup.sh first)"
[ "$(code "$GITLAB")" = 200 ] || die "GitLab is not returning 200 before any change; not touching anything"
podman container exists nginx || die "no container named nginx"
podman exec nginx nginx -t -q || die "the proxy's current config does not pass nginx -t; not touching it"
getent hosts "$DOMAIN" >/dev/null || die "$DOMAIN does not resolve yet (add its DNS record first)"
podman exec nginx wget -q -O /dev/null -T 5 "http://$UPSTREAM/api/state" \
  || die "the proxy cannot reach the dashboard at $UPSTREAM (is './panther dashboard' running? a firewall in the way?)"
for f in panther-http.conf panther.conf; do
  [ ! -e "$CONF/$f" ] || die "$CONF/$f already exists; refusing to overwrite it"
done

# --- 1. port 80, for the certificate challenge ---
if [ "$DRY" = 1 ]; then echo "+ render panther-http.conf -> $CONF/panther-http.conf"; else render panther-http.conf "$CONF/panther-http.conf"; fi
record "rm -f $CONF/panther-http.conf && $RELOAD"
sh_ "$RELOAD"

# --- 2. the certificate ---
do_ "${CERTBOT[@]}" certonly --webroot -w /var/www/certbot -d "$DOMAIN" --cert-name panther \
  --non-interactive --agree-tos --register-unsafely-without-email --keep-until-expiring
record "${CERTBOT[*]} delete --cert-name panther --non-interactive"

# --- 3. HTTPS ---
if [ "$DRY" = 1 ]; then echo "+ render panther.conf -> $CONF/panther.conf"; else render panther.conf "$CONF/panther.conf"; fi
record "rm -f $CONF/panther.conf && $RELOAD"
sh_ "$RELOAD"

# --- post-checks ---
if [ "$DRY" != 1 ]; then
  sleep 2
  # Through the proxy on this host, with the real name and certificate: the server can't reach its own public
  # address (no hairpin NAT), so a plain request for the domain from here fails even when the site works.
  local_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 --resolve "$DOMAIN:443:127.0.0.1" "https://$DOMAIN/" || true)"
  [ "$local_code" = 200 ] || { echo "proxy-setup: https://$DOMAIN/ via this host's proxy is $local_code, not 200" >&2; false; }
  [ "$(code "$GITLAB")" = 200 ] || { echo "proxy-setup: GitLab not 200 after changes" >&2; false; }
fi

trap - ERR
echo "proxy-setup: done: https://$DOMAIN/  Ledger: $LEDGER"
[ "$DRY" = 1 ] || cat "$LEDGER"
