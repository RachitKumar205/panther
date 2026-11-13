# pi

The Raspberry Pi 5 only checks our winners (never presets or other baselines): real peak RAM against the
prediction, and speed.

| File | What |
|---|---|
| `setup.sh` | Builds llama.cpp at the pinned commit (after `sudo apt-get install -y git cmake build-essential python3`) |
| `await.sh` | Waits for each winner's Hugging Face repo, downloads it, checks it, uploads the result line to the status page, deletes the model file |
| `panther_pi.py` | The check: checksum, llama-server at 4,096 tokens with an 8-bit KV cache, 256-token prompt + 64 generated, peak RAM |
| `lean.sh` | Stops services the check does not need, for RAM (a reboot restores them) |

`nohup bash pi/await.sh <hf-user> 5:14b 7:24b > ~/panther-pi/await.log 2>&1 &`
