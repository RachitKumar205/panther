"""Quick Pi check (D44) end to end, against a fake llama-server binary."""

import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
import panther_pi as pp  # noqa: E402

FAKE = r'''#!PY
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
a = sys.argv
model, port = a[a.index("-m") + 1], int(a[a.index("--port") + 1])
print("llama_model_loader: - type  f32:    3 tensors", flush=True)
if "14b" in model:
    sys.exit(1)  # simulate a load failure (e.g. out of memory)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            out = json.dumps({"tokens": list(range(len(body["content"].split())))}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(out); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        n = body["n_predict"]
        for i in range(n):
            self.wfile.write(("data: " + json.dumps({"content": "t ", "stop": False}) + "\n\n").encode())
        t = {"prompt_n": len(body["prompt"]), "prompt_per_second": 20.0, "predicted_n": n, "predicted_per_second": 3.0}
        self.wfile.write(("data: " + json.dumps({"content": "", "stop": True, "timings": t}) + "\n\n").encode())
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def make_bundle(tmp: Path, recommended: str, files: list[str]) -> tuple[Path, Path]:
    llama = tmp / "llama.cpp"
    binp = llama / "build" / "bin" / "llama-server"
    binp.parent.mkdir(parents=True)
    binp.write_text(FAKE.replace("PY", sys.executable, 1))
    binp.chmod(binp.stat().st_mode | stat.S_IEXEC)
    bundle = tmp / "bundle"
    bundle.mkdir()
    (bundle / "agreement.txt").write_text(" ".join(["w"] * 400))
    meta = {}
    for name in ["qwen2.5-7b-poc-winner.gguf", "qwen2.5-7b-poc-uniform.gguf",
                 "qwen2.5-14b-poc-winner.gguf", "qwen2.5-14b-poc-uniform.gguf"]:
        content = name.encode()
        if name in files:
            (bundle / name).write_bytes(content)
        import hashlib
        meta[name] = {"model": "Qwen2.5-14B-Instruct" if "14b" in name else "Qwen2.5-7B-Instruct",
                      "role": "winner" if "winner" in name else "uniform", "sha256": hashlib.sha256(content).hexdigest(),
                      "predicted_ram_gib": 0.01, "predicted": {"weights": 0, "kv_cache": 0},
                      "expected_type_counts": {"f32": 3}}
    manifest = {"files": meta, "agreement_text": "agreement.txt", "recommended_file": recommended,
                "recommended_reason": "test", "verdict": {"comparisons": {
                    "7B winner vs 14B winner": {"winner": "14B:top4"},
                    "7B winner vs uniform": {"winner": "7B:uniform"},
                    "14B winner vs uniform": {"winner": "14B:top4"}}}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle, llama


def runs(bundle: Path) -> list[dict]:
    return [json.loads(x) for x in (bundle / "results" / "runs.jsonl").read_text().splitlines()]


def test_fallback_follows_verdict(tmp_path):
    bundle, _ = make_bundle(tmp_path, "qwen2.5-14b-poc-winner.gguf", [])
    m = json.loads((bundle / "manifest.json").read_text())
    assert pp.fallback_file(m) == "qwen2.5-7b-poc-uniform.gguf"


def test_quick_measures_only_the_recommended_file(tmp_path):
    bundle, llama = make_bundle(tmp_path, "qwen2.5-7b-poc-winner.gguf",
                                ["qwen2.5-7b-poc-winner.gguf", "qwen2.5-14b-poc-winner.gguf"])
    assert pp.main(["run", "--quick", "--bundle", str(bundle), "--llama", str(llama)]) == 0
    r = runs(bundle)
    assert [x["run_key"] for x in r] == ["verify:qwen2.5-7b-poc-winner.gguf", "Q:qwen2.5-7b-poc-winner.gguf:t4:256/64"]
    assert r[1]["valid"] and len(r[1]["runs"]) == 3
    rep = json.loads((bundle / "results" / "report.json").read_text())
    assert rep["recommended_confirmed_on_pi"] is True


def test_quick_falls_back_when_recommended_fails(tmp_path):
    bundle, llama = make_bundle(tmp_path, "qwen2.5-14b-poc-winner.gguf",
                                ["qwen2.5-14b-poc-winner.gguf", "qwen2.5-7b-poc-uniform.gguf"])
    assert pp.main(["run", "--quick", "--bundle", str(bundle), "--llama", str(llama)]) == 0
    keys = [x["run_key"] for x in runs(bundle)]
    assert "Q:qwen2.5-7b-poc-uniform.gguf:t4:256/64" in keys
    assert not any(k.startswith("Q:qwen2.5-14b") for k in keys)
    rep = json.loads((bundle / "results" / "report.json").read_text())
    assert rep["recommended_confirmed_on_pi"] is False
    assert "FAILED" in (bundle / "results" / "queue.log").read_text()
