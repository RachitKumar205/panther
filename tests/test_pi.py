"""Pi measurement tool against a mock llama-server (streams tokens like the real one)."""

import json
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
    assert pp.parse_final_ppl("x\nFinal estimate: PPL = 9.1234 +/- 0.2\n") == 9.1234
    rec = {"peak_ram_gib": 7.2, "min_mem_available_mib": 100, "temp_start_c": 50.0, "throttled_start": "0x0",
           "throttled_end": "0x50005", "type_counts": {"f32": 1}, "expected_type_counts": {"f32": 2},
           "workload": [128, 32], "runs": [{"predicted_n": 32}]}
    ok, reasons = pp.validity(rec)
    assert not ok and len(reasons) == 4
    assert pp.percentile([1, 2, 3, 4], 0.5) == 2.5
