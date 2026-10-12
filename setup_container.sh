#!/usr/bin/env bash
# Run inside the pinned container from README, not directly on the host.
set -euo pipefail
git clone --depth 1 --branch v0.6.1 https://github.com/verl-project/verl.git /opt/verl-src
test "$(git -C /opt/verl-src rev-parse HEAD)" = d62da4950573d7a4b7ef2362337952e7ab59e78d
python -m pip install --no-deps -e /opt/verl-src
mkdir -p runs
python -m pip freeze > runs/environment.txt
python -c 'import torch, transformers, vllm, verl; print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__); print("CUDA", torch.version.cuda, "GPUs", torch.cuda.device_count()); assert vllm.__version__ == "0.11.0"; assert torch.cuda.device_count() >= 2; assert torch.cuda.is_bf16_supported()'
