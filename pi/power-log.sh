#!/usr/bin/env bash
# Run on the Pi during a check: every 5 s, one line of temperature, input voltage, throttle flags and free RAM,
# synced to disk so it survives a power cut.
#
#   bash pi/power-log.sh > ~/panther-pi/power.log &
while :; do
  echo "$(date +%T) $(vcgencmd measure_temp) $(vcgencmd pmic_read_adc EXT5V_V | awk '{print $2}')" \
       "$(vcgencmd get_throttled) avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)MB"
  sync
  sleep 5
done
