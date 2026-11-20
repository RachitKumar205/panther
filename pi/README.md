# pi

The Raspberry Pi 5 only checks our winners (never presets or other baselines): real peak RAM against the
prediction and the budget, and speed.

| File | What |
|---|---|
| `setup.sh` | Builds llama.cpp's server at the pinned commit (`~/panther-pi/llama.cpp`) and installs the Hugging Face CLI (`~/hf-venv`, `huggingface_hub` at the version in `uv.lock`) |
| `panther_pi.py` | The queue: waits for each winner's Hugging Face repo, downloads it, checks it, posts the result line, deletes the model file |

## Setup (once)

```bash
sudo apt-get install -y git cmake build-essential python3 python3-venv   # the Pi's admin; the only sudo step
bash pi/setup.sh                                                          # as an ordinary user
~/hf-venv/bin/hf auth login                                               # a write token: results are uploaded
```

## Running

```bash
nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &
```

For each contest (`<size>@<budget>`) it polls `<hf-user>/panther-<size>-<budget>g-v2` every 5 minutes until the
winner is there, downloads it to `~/panther-pi/budget-<budget>-<size>-v2/`, and checks it: the file's checksum
against the manifest, then llama-server at the bundle's context with an 8-bit KV cache, flash attention, 4 threads
and weight repacking off (on ARM, repacking keeps a second copy of K-type weights: the 7B @ 5 GiB winner peaked at
6.86 GiB with it, 4.69 without), a 256-token prompt and 64 generated tokens. The result line, e.g.

```
14:02:11 Q <file> t=4 (256, 64): decode 3.21 t/s, prefill 18.40 t/s, peak RAM 4.690 GiB (predicted 4.712), FITS 5 GiB, throttled: throttled=0x0 -> throttled=0x0
```

has decode and prefill speed, the server's real peak RAM (VmHWM) and the prediction, whether that peak is within the
budget (`FITS`/`OVER <budget> GiB`; reported only, the verdict does not depend on it), and the Pi's throttling flags
before and after. It is appended to `results/queue.log` in the bundle folder, written to `pi-result.txt`, printed,
and uploaded to `<hf-user>/panther-status` as `pi/<contest>.txt` (the dashboard reads it there). Then the `.gguf`
is deleted (the SD card holds one model at a time) and the next contest is checked.

## A failed check

A checksum mismatch, a server that exits before it is ready (e.g. killed for lack of memory), one that is not
healthy after 900 tries (about 15 minutes), or one that stops answering: that contest's result line is
`HH:MM:SS FAILED: <the error>`, posted exactly like a result (queue log, `pi-result.txt`, upload), printed on
stderr as `FAILED: <contest>: <error>` with the traceback, and the server's output stays in `server.log` in the
bundle folder. The `.gguf` is deleted and the queue goes on. To recheck a contest after fixing it, start the queue
again with that contest. A failed upload (e.g. no write token) stops the queue with the error, before the model
file is deleted.
