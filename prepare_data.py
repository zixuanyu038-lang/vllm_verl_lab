"""Deterministic, disjoint arithmetic data; no model or GPU needed for JSONL."""
import argparse
import json
import random
from pathlib import Path


def build_rows():
    rng = random.Random(20261012)
    cases = set()
    while len(cases) < 768:
        a, b = sorted((rng.randrange(10, 50), rng.randrange(10, 50)))
        cases.add((a, b, rng.randrange(0, 100)))
    cases = sorted(cases)
    rng.shuffle(cases)
    rows = []
    for i, (a, b, c) in enumerate(cases):
        question = f"计算 ({a} × {b}) + {c}。只输出最终答案，格式为 #### 整数。"
        truth = str(a * b + c)
        rows.append({"id": i, "case": [a, b, c],
                     "prompt": [{"role": "user", "content": question}],
                     "answer": f"#### {truth}", "ground_truth": truth})
    return {"train": rows[:512], "val": rows[512:640], "test": rows[640:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for split, rows in build_rows().items():
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            converted = [{"data_source": "toy_arithmetic", "prompt": x["prompt"],
                          "ability": "math",
                          "reward_model": {"style": "rule", "ground_truth": x["ground_truth"]},
                          "extra_info": {"split": split, "index": x["id"]}} for x in rows]
            pq.write_table(pa.Table.from_pylist(converted), target / f"{split}.parquet")
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
