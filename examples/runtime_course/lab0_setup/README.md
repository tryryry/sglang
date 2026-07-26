# 实验 0:环境准备(约 30 分钟)

## 目标

搭建可运行 SGLang 的环境,并验证 CUDA / PyTorch / sgl-kernel 均正常。

## 步骤

### 1. 安装 SGLang

推荐使用 uv 加速安装:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]"
```

或者使用官方 Docker 镜像(无本地 GPU 环境者推荐,详见 [docs/get_started/install.md](../../../docs/get_started/install.md)):

```bash
docker run --gpus all -it --shm-size 32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    lmsysorg/sglang:latest bash
```

### 2. 验证环境

```bash
python -m sglang.check_env
```

确认输出中以下项正常:

- `CUDA available: True`,且 CUDA / Driver 版本匹配
- `PyTorch` 版本
- `sglang` 与 `sgl-kernel` 版本
- GPU 型号与显存大小

### 3. 预下载模型权重

提前下载小模型,避免后续实验时等待:

```bash
pip install "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-0.5B-Instruct
```

## 产出

- 环境自检报告截图(`python -m sglang.check_env` 的完整输出)。

## 思考题

1. `check_env` 报告的 CUDA 版本与 `nvidia-smi` 显示的版本有什么区别?
2. 为什么 Docker 启动时需要 `--shm-size`?(提示:多进程间张量传递)
