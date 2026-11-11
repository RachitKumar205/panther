# Raspberry Pi 5 session (proof of concept, Phase E)

What the Pi does: confirm that the configurations chosen on the server fit in the RAM budget (7.0 GiB) and
measure their real speed. It does not search; the server already chose. Plan and rules: `docs/PIPELINE.md`
(Phase E for the proof of concept) and `docs/DECISIONS.md` D42.

Everything runs as your normal user. The only `sudo` is installing build tools once.

## 0. On your Mac: is the server side finished?

Open the server's SSH control master (as usual), then:

```bash
deploy/server/wait-job.sh bundle 1
```

`panther-bundle: exited 0` means the bundle is ready. Anything else: the chains stopped somewhere; see
`data/logs/chain-*.log` on the server (`chain-7b`, `chain-14b`, `chain-final`).

## 1. On the Pi: build tools and llama.cpp (~30 min, once)

```bash
sudo apt-get install -y git cmake build-essential python3
git clone https://github.com/RachitKumar205/panther ~/panther
bash ~/panther/pi/setup.sh
```

## 2. On your Mac: copy the bundle to the Pi (~25 GB)

```bash
pi/fetch_bundle.sh pi@<pi-hostname>
```

This copies server → Mac (`~/panther-bundle`) → Pi (`~/panther-pi/bundle`).

## 3. On the Pi: run the queue (~3–4 h, unattended)

Use `tmux` or `nohup` so it survives disconnecting:

```bash
cd ~/panther && nohup python3 pi/panther_pi.py run > ~/panther-pi/queue.out 2>&1 &
tail -f ~/panther-pi/bundle/results/queue.log
```

It checks every file's sha256, then: loads each file once and checks peak RAM against the prediction; tries
2/3/4 threads; runs the 256/64-token confirmation on all four files; checks perplexity agreement with the
server; runs 128/32 and 1024/128 on the recommended file. It waits for the CPU to cool below 70 °C between
files. If it is interrupted, run the same command again: finished runs are skipped.

Keep the Pi otherwise idle while it runs (no desktop session if possible): background load spoils speed and
`MemAvailable` numbers.

## 4. Results

`~/panther-pi/bundle/results/report.md` (and `report.json`, `runs.jsonl`, logs). Copy it back with

```bash
scp pi@<pi-hostname>:panther-pi/bundle/results/report.md .
```
