# Observability

## Dashboard

- Read-only, served from `data/` by `panther-dashboard`.
- Endpoints (contract, ADR 0007): `/api/state`, `/api/logs`, `/api/logs/stream`.
- Masks the server account/address and the shared service's domain in everything it sends.
- At most 16 live log streams; the page polls state every 10 s.
- Pages: Overview, Contests, Contest (Results / Steps / Events), System, Logs.

## Raspberry Pi

- The queue downloads each winner bundle, verifies its SHA-256 against the manifest, runs
  `llama-server` at the bundle's context/KV type with repacking off, and reports:
  decode/prefill speed, peak RAM (VmHWM), predicted RAM, `FITS`/`OVER`, throttling.
- Results are written to `pi-result.txt` and uploaded as `pi/<contest>.txt` to
  `<user>/panther-status`.
- A failed check posts `HH:MM:SS FAILED: <error>` and the queue continues.
