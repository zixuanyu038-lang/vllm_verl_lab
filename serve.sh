#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec vllm serve "$MODEL_PATH" --served-model-name toy \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size "${TP:-1}" --max-model-len 512 \
  --gpu-memory-utilization 0.5 --enforce-eager \
  --no-enable-prefix-caching --generation-config vllm
