#!/usr/bin/env bash
# Run on the Pi before a check: stop what the check doesn't need, so the model gets as much RAM as possible.
# Keeps SSH, networking, Tailscale and rpi-connect. Nothing is disabled: a reboot brings everything back.
#
#   bash pi/lean.sh
set -uo pipefail
avail() { awk '/MemAvailable/ {printf "%.2f GiB", $2 / 1048576}' /proc/meminfo; }
echo "available before: $(avail)"

sudo systemctl isolate multi-user.target                                 # the desktop
sudo systemctl stop cloud-init-main.service bluetooth.service \
  avahi-daemon.service avahi-daemon.socket                               # first-boot setup, Bluetooth, discovery
systemctl --user stop pipewire.socket pipewire-pulse.socket pipewire pipewire-pulse wireplumber \
  xdg-desktop-portal gvfs-udisks2-volume-monitor gvfs-daemon 2>/dev/null # sound, desktop helpers

sleep 2
echo "available after:  $(avail)"
