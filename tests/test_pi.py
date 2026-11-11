"""Pi measurement tool against a mock llama-server (streams tokens like the real one)."""

import json
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
import panther_pi as pp  # noqa: E402


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            out = json.dumps({"tokens": list(range(len(body["content"].split())))}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = body["n_predict"]
        for i in range(n):
            time.sleep(0.01)
            msg = {"content": f"t{i} ", "stop": False}
            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
            self.wfile.flush()
        final = {"content": "", "stop": True, "timings": {"prompt_n": len(body["prompt"]), "prompt_per_second": 50.0,
                                                          "predicted_n": n, "predicted_per_second": 90.0}}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())


def test_generate_and_stats_against_mock_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    srv = pp.Server(Path("/nonexistent"), Path("m.gguf"), 4, Path("/dev/null"))
    srv.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    ids = srv.tokenize("a b c d e f")
    assert ids == [0, 1, 2, 3, 4, 5]
    g = srv.generate(ids[:4], 12)
    st = pp.run_stats(g)
    httpd.shutdown()
    assert st["tokens_streamed"] == 12 and st["predicted_n"] == 12 and st["prompt_n"] == 4
    assert st["decode_tps"] == 90.0 and st["itl_p50_ms"] > 5 and st["itl_p95_ms"] >= st["itl_p50_ms"]


def test_parsers_and_validity():
    log = "llama_model_loader: - type  f32:  141 tensors\nllama_model_loader: - type q6_K:  196 tensors\n"
    assert pp.parse_type_counts(log) == {"f32": 141, "q6_k": 196}
    rec = {"peak_ram_gib": 7.2, "min_mem_available_mib": 100, "temp_start_c": 50.0, "throttled_start": "0x0",
           "throttled_end": "0x50005", "type_counts": {"f32": 1}, "expected_type_counts": {"f32": 2},
           "workload": [128, 32], "runs": [{"predicted_n": 32}]}
    ok, reasons = pp.validity(rec)
    assert not ok and len(reasons) == 4
    assert pp.percentile([1, 2, 3, 4], 0.5) == 2.5


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


def test_check_measures_every_file(tmp_path):
    bundle, llama = make_bundle(tmp_path, "qwen2.5-7b-poc-winner.gguf", ["qwen2.5-7b-poc-winner.gguf"])
    m = json.loads((bundle / "manifest.json").read_text())
    m["files"] = {k: v for k, v in m["files"].items() if k == "qwen2.5-7b-poc-winner.gguf"}
    (bundle / "manifest.json").write_text(json.dumps(m))
    assert pp.main(["--bundle", str(bundle), "--llama", str(llama)]) == 0
    r = runs(bundle)
    assert [x["run_key"] for x in r] == ["verify:qwen2.5-7b-poc-winner.gguf", "Q:qwen2.5-7b-poc-winner.gguf:t4:256/64"]
    assert r[1]["valid"] and len(r[1]["runs"]) == 3
    assert "qwen2.5-7b-poc-winner.gguf" in (bundle / "results" / "report.md").read_text()


def test_check_reports_a_failed_load(tmp_path):
    bundle, llama = make_bundle(tmp_path, "qwen2.5-14b-poc-winner.gguf", ["qwen2.5-14b-poc-winner.gguf"])
    m = json.loads((bundle / "manifest.json").read_text())
    m["files"] = {k: v for k, v in m["files"].items() if k == "qwen2.5-14b-poc-winner.gguf"}
    (bundle / "manifest.json").write_text(json.dumps(m))
    assert pp.main(["--bundle", str(bundle), "--llama", str(llama)]) == 0
    assert "FAILED" in (bundle / "results" / "queue.log").read_text()
