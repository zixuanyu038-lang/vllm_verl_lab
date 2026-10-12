#!/usr/bin/env bash
set -euo pipefail
python -m verl.model_merger merge --backend fsdp \
  --local_dir runs/grpo/global_step_20/actor --target_dir runs/grpo_hf
# Explicitly retain the same tokenizer/chat template as SFT.
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("runs/sft").save_pretrained("runs/grpo_hf")'
