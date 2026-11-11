#!/usr/bin/env bash
# Run on the Pi before a check: stop what the check doesn't need, so the model gets as much RAM as possible.
# Keeps SSH, networking, Tailscale and rpi-connect. Nothing is disabled: a reboot brings everything back.
# An optional MHz caps the CPU clock (normal maximum 2400), for a power supply that can't hold full load;
# speeds measured under a cap are lower than the Pi's real ones.
#
#   bash pi/lean.sh          # stop the extras
#   bash pi/lean.sh 1800     # and cap the CPU at 1.8 GHz until the next reboot
set -uo pipefail
avail() { awk '/MemAvailable/ {printf "%.2f GiB", $2 / 1048576}' /proc/meminfo; }
echo "available before: $(avail)"

sudo systemctl isolate multi-user.target                                 # the desktop
sudo systemctl stop cloud-init-main.service bluetooth.service \
  avahi-daemon.service avahi-daemon.socket                               # first-boot setup, Bluetooth, discovery
systemctl --user stop pipewire.socket pipewire-pulse.socket pipewire pipewire-pulse wireplumber \
  xdg-desktop-portal gvfs-udisks2-volume-monitor gvfs-daemon 2>/dev/null # sound, desktop helpers

if [ -n "${1:-}" ]; then
  echo "$(( $1 * 1000 ))" | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq >/dev/null
  echo "CPU capped at $1 MHz until the next reboot"
fi
sleep 2
echo "available after:  $(avail)"
