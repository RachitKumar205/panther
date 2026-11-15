# pi

The Raspberry Pi 5 only checks our winners (never presets or other baselines): real peak RAM against the
prediction, and speed.

| File | What |
|---|---|
| `setup.sh` | Builds llama.cpp at the pinned commit (after `sudo apt-get install -y git cmake build-essential python3`) |
| `panther_pi.py` | The Pi's one command: waits for each winner's Hugging Face repo, downloads it, checks it (checksum; llama-server at 4,096 tokens with an 8-bit KV cache; a 256-token prompt and 64 generated tokens; peak RAM), posts the result line to the live page, deletes the model file |
| `runtime_check.py` | The same check of one bundle with a chosen server binary and options, to compare runtimes (mainline llama.cpp, ik_llama.cpp) on the same file; keeps the file, appends to `results/runtimes.log` |
| `membw.c` | The Pi's memory read bandwidth with 1 to N threads (`gcc -O3 -march=native -pthread pi/membw.c -o membw && ./membw 2048 4`): the ceiling for writing tokens, since each token reads every weight once |

`nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`
