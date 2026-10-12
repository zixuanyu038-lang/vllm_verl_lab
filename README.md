# 双卡 Linux vLLM SFT GRPO 实验

与两篇公众号实验配套。WSL 仅作为终端和 SSH 入口，正式实验在一台 Linux NVIDIA 双卡主机完成。

**这里提供可运行配方，不代表已经完成双卡 GPU 实测。synthetic_results.json 是明确标注的模拟数据，不由训练脚本产生。真实脚本将结果写到 runs/，不读取模拟文件。**

## 环境

- 主示例：2 × RTX 3090 24GB。不是显存最低要求或性能保证。
- 宿主：Linux x86_64，NVIDIA 驱动、Docker、NVIDIA Container Toolkit 已配置。
- 保守驱动基线：Linux 570.124.06 或更新（CUDA 12.8.1 对应驱动），不是理论最低兼容版本。
- 固定镜像：verlai/verl:vllm011.dev7，digest 见下。
- 固定源码：verl v0.6.1，commit d62da4950573d7a4b7ef2362337952e7ab59e78d。
- vLLM 0.11.0；其余编译依赖沿用固定镜像，不单独升级。
- 模型：Qwen/Qwen2.5-0.5B-Instruct，revision 7ae557604adf67be50417f59c2c2f167def9a775；下载时记录到 models/model_revision.json。

所有命令从本目录执行，脚本是 LF 换行。主机驱动和容器依赖是不同层，容器不会安装或替代宿主驱动。

## 1 启动容器

在 Linux 主机、本目录执行：

```bash
IMAGE='verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d'
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace/lab" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/lab --entrypoint /bin/bash "$IMAGE"
```

在容器内执行：

```bash
bash setup_container.sh
python download_model.py
python prepare_data.py --parquet
```

安装脚本使用 --no-deps，依赖沿用镜像。第一次需要网络和足够的磁盘空间下载镜像、模型。不要在此环境运行 pip install -U torch/vllm/flash-attn。

第二个终端可以在宿主运行 docker ps 找到 ID，然后 docker exec -it <ID> bash，进入后 cd /workspace/lab。容器用 Linux host 网络，127.0.0.1 可从同一主机访问；不要换成 -p 映射却保留容器内 loopback 监听。

## 2 基础模型推理与评分

终端 A（容器内）：

```bash
bash serve.sh models/qwen-0.5b
```

终端 B（同一容器）：

```bash
curl -f http://127.0.0.1:8000/health
python bench.py --concurrency 1 --out runs/bench_c1.json
python bench.py --concurrency 8 --out runs/bench_c8.json
python evaluate.py --data data/val.jsonl --out runs/base_val.json
```

serve 默认 GPU0、TP1、512总上下文、50%显存预算、eager、前缀缓存关闭、generation-config vllm。bench 固定输入、32请求、64输出上限、ignore_eos=True、预热排除，统计客户端首段文本延迟和总输出/墙钟时间；不是生产负载基准。evaluate 不忽略EOS，temperature0，128输出上限，输出原文和严格答案/格式评分。

TP2实验先停止旧服务，然后：

```bash
CUDA_VISIBLE_DEVICES=0,1 TP=2 bash serve.sh models/qwen-0.5b
```

本模型14个query头，TP4不适用。

## 3 SFT

停止所有独立 vLLM 服务并确认显存释放，在容器本目录执行：

```bash
bash sft.sh
```

全参数 DDP，两进程，每卡microbatch2、累积4、有效batch16；40个optimizer step。只监督assistant回答（含结束标记），prompt/padding labels为-100。完整 HF 权重保存到 runs/sft。

启动服务 bash serve.sh runs/sft，然后另一个终端：

```bash
python evaluate.py --data data/val.jsonl --out runs/sft_val.json
```

## 4 GRPO

停止独立服务并确认显存释放，再在容器本目录：

```bash
mkdir -p runs
set -o pipefail
bash grpo.sh 2>&1 | tee runs/grpo.log
```

FSDP训练，rollout TP1，16prompts×4responses。v0.6.1旧FSDP worker会将mini16乘n4：全局64responses/minibatch，两卡每卡32，micro1累积32次后更新；每轮1次actor optimizer step，总20轮。参考策略、critic、GPU reward model均不启用；这不是长期训练的稳定性配置。

文件输出：runs/grpo/global_step_10/actor、global_step_20/actor 分片；runs/rollouts、runs/validation 原始生成；runs/grpo.log 控制台日志。

验证核心指标为 val-core/toy_arithmetic/reward/mean@1，日志细节以固定版本输出为准。score是严格答案评分，format_ok只用于排查；同组奖励相同则GRPO优势为0，不保证有改善。

## 5 合并与最终测试

```bash
bash merge_grpo.sh
bash serve.sh runs/grpo_hf
```

在另一个容器终端：

```bash
python evaluate.py --data data/val.jsonl --out runs/grpo_val.json
python evaluate.py --data data/test.jsonl --out runs/grpo_test.json
```

最终确定配置后，用基础模型和SFT模型分别启动服务，再运行相同test命令，输出分别改为runs/base_test.json、runs/sft_test.json。每次先停止上一个服务，不覆盖旧结果。val参与调参，test保留到最后。

## 本地可做的检查

```bash
python -m compileall -q .
python check_cpu.py
```

check_cpu只检查数据划分、评分边界和HTTP/SSE客户端协议，用假服务响应检查工具；不是GPU推理或训练测试。GPU环境、模型tokenizer、DDP/FSDP/NCCL、速度和训练效果仍需在目标机器运行。

正式复现请保留模型revision、runs/environment.txt、nvidia-smi输出、启动命令和原始日志，避免把模拟摘要混入实测记录。
