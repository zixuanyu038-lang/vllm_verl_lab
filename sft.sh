#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
torchrun --standalone --nnodes=1 --nproc-per-node=2 train_sft.py \
  --model "${MODEL_PATH:-models/qwen-0.5b}" \
  --data data/train.jsonl --out runs/sft --steps 40
