"""Small HTTP concurrency probe, not a replacement for vllm bench serve."""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def request_one(base, model, output_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": "请介绍矩阵乘法。"}],
               "temperature": 0, "max_tokens": output_tokens,
               "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            text = raw[6:].decode("utf-8").strip()
            if text == "[DONE]":
                break
            chunk = json.loads(text)
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content") and first is None:
                    first = time.perf_counter()
            if chunk.get("usage"):
                usage = chunk["usage"]
    if first is None or usage is None:
        raise RuntimeError("Missing streamed text or usage; cannot compute the requested metrics")
    return {"ttft_s": first - started, "e2e_s": time.perf_counter() - started,
            "output_tokens": usage["completion_tokens"], "input_tokens": usage["prompt_tokens"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="toy")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--out", default="runs/bench.json")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.output_tokens) < 1:
        parser.error("requests, concurrency and output-tokens must be positive")
    request_one(args.base, args.model, args.output_tokens)  # excluded warmup
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda _: request_one(args.base, args.model, args.output_tokens), range(args.requests)))
    elapsed = time.perf_counter() - start
    total = sum(x["output_tokens"] for x in rows)
    report = {"source": "measured_http", "requests": args.requests,
              "concurrency": args.concurrency, "wall_s": elapsed,
              "output_tokens": total, "output_tokens_per_s": total / elapsed,
              "ttft_mean_s": statistics.mean(x["ttft_s"] for x in rows), "rows": rows}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
