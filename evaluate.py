"""Evaluate a local OpenAI-compatible endpoint with the same rule as GRPO."""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from reward import compute_score


def complete(base, model, messages):
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 128}
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    return body["choices"][0]["message"]["content"] or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="toy")
    parser.add_argument("--data", default="data/val.jsonl")
    parser.add_argument("--out", default="runs/eval.json")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines()]
    if not rows:
        parser.error("empty dataset")
    def score(row):
        answer = complete(args.base, args.model, row["prompt"])
        return {"id": row["id"], "response": answer,
                **compute_score("toy_arithmetic", answer, row["ground_truth"])}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(score, rows))
    report = {"source": "measured_http", "model": args.model, "data": args.data,
              "count": len(results), "correct": int(sum(x["score"] for x in results)),
              "format_ok": int(sum(x["format_ok"] for x in results)), "rows": results}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
