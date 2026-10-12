"""CPU-only checks, including a deliberately fake OpenAI-compatible HTTP server."""
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys

from bench import request_one
from evaluate import complete
from prepare_data import build_rows
from reward import compute_score


def main():
    splits = build_rows()
    assert [len(splits[x]) for x in ["train", "val", "test"]] == [512, 128, 128]
    cases = [tuple(row["case"]) for rows in splits.values() for row in rows]
    assert len(set(cases)) == 768
    for rows in splits.values():
        for row in rows:
            a, b, c = row["case"]
            assert int(row["ground_truth"]) == a * b + c
            assert compute_score("toy", row["answer"], row["ground_truth"])["score"] == 1
    assert compute_score("toy", "#### 936", "936")["score"] == 1
    assert compute_score("toy", "#### 937", "936")["score"] == 0
    assert compute_score("toy", "答案是936", "936")["score"] == 0
    assert compute_score("toy", "#### 936 or #### 937", "936")["score"] == 0
    assert compute_score("toy", "#### -3", "-3")["score"] == 1

    class FakeEndpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if payload.get("stream"):
                assert payload["stream_options"]["include_usage"] is True
                assert payload["ignore_eos"] is True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                    {"choices": [{"delta": {"content": "假响应，"}}]},
                    {"choices": [{"delta": {"content": "协议检查"}}]},
                    {"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 13}},
                ]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "#### 936"}}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        assert complete(base, "toy", [{"role": "user", "content": "test"}]) == "#### 936"
        result = request_one(base, "toy", 64)
        assert result["output_tokens"] == 7  # usage, not the number of SSE chunks
        assert result["input_tokens"] == 13
        assert 0 <= result["ttft_s"] <= result["e2e_s"]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fake_bench.json"
            subprocess.run([sys.executable, str(Path(__file__).with_name("bench.py")),
                            "--base", base, "--requests", "4", "--concurrency", "2",
                            "--out", str(target)], check=True, capture_output=True)
            report = json.loads(target.read_text(encoding="utf-8"))
            assert report["output_tokens"] == 28
            assert report["requests"] == len(report["rows"]) == 4
            assert report["output_tokens_per_s"] == 28 / report["wall_s"]
    finally:
        server.shutdown()
        server.server_close()
    synthetic = json.loads(Path(__file__).with_name("synthetic_results.json").read_text(encoding="utf-8"))
    assert synthetic["source"] == "SIMULATED_NOT_MEASURED"
    for row in synthetic["benchmark"]:
        assert abs(row["output_tokens_per_s"] - row["output_tokens"] / row["wall_s"]) < 1e-9
    print("PASS: CPU data/reward/HTTP/SSE checks; fake endpoint only, no GPU inference or training")


if __name__ == "__main__":
    main()
