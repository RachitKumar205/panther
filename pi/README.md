# pi

The Raspberry Pi 5 (8 GB) side. The Pi measures only peak RAM and speed of **our** winners; existing quantizations
(llama.cpp presets, uniform files) never run on it.

| File | What |
|---|---|
| `setup.sh` | One-time setup: llama.cpp at the pinned commit (CPU build), Python environment, `hf` CLI |
| `await.sh` | Waits for each winner's private Hugging Face repo, downloads it and runs the check, one at a time; needs neither the Mac nor the server. `nohup bash pi/await.sh <hf-user> 7:24b 7:14b > ~/panther-pi/await.log &` |
| `panther_pi.py` | The check: starts llama-server on the bundle, measures peak RAM (sampled), prompt and writing speed (256-token prompt, 64 tokens written, 4 threads, 4,096-token context, 8-bit KV cache, 1 warm-up + 2 timed runs), and the throttle flags |
| `lean.sh` | Stops the desktop and other extras before a check (keeps SSH, networking, Tailscale and rpi-connect); nothing persists past a reboot. `bash pi/lean.sh 1800` also caps the CPU at 1.8 GHz |
| `power-log.sh` | Temperature, input voltage, throttle flags and free RAM every 5 s, synced to disk |

Network: the phone hotspot `jia` (priority 20) when it is on, the university network `Student` (priority 10)
otherwise, also at boot. The Pi has no working HDMI output.
