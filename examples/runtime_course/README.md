# SGLang Runtime 实战课程实验(Hands-on Course Labs)

本课程带你从零跑起 SGLang,并逐步深入 Runtime(SRT)核心机制。

## 环境要求

- 一张 NVIDIA GPU(建议 ≥16GB 显存,如 A10 / 3090 / 4090 / A100),Python ≥ 3.10
- 无本地 GPU 者可使用官方 Docker 镜像 `lmsysorg/sglang` 在云主机上完成
- 全程使用小模型 `Qwen/Qwen2.5-0.5B-Instruct` 或 `meta-llama/Llama-3.2-1B-Instruct` 降低门槛

## 实验目录

| 实验 | 主题 | 预计用时 | 必做 |
|---|---|---|---|
| [实验 0](./lab0_setup/README.md) | 环境准备 | 30 分钟 | ✅ |
| [实验 1](./lab1_first_run/README.md) | 第一次跑起来:启动服务器并发请求 | 1 小时 | ✅ |
| [实验 2](./lab2_offline_engine/README.md) | 离线 Engine API:不起服务器直接推理 | 1 小时 | ✅ |
| [实验 3](./lab3_radix_cache/README.md) | RadixAttention 前缀缓存实测 | 1.5 小时 | ✅ |
| [实验 4](./lab4_scheduler/README.md) | 连续批处理与调度器观察 | 1.5 小时 | ✅ |
| [实验 5](./lab5_code_walkthrough/README.md) | 走读一条请求的代码链路 | 2 小时 | ✅ |
| [实验 6](./lab6_advanced/README.md) | 进阶:高级特性体验 | 2 小时 | 选做 |

## 考核建议

- 实验 1–5 为必做,每个实验提交产出物 + 思考题答案。
- 期末小项目(二选一):
  1. 基于 Engine API 搭一个带前缀缓存优化意识的多轮对话应用,并给出压测报告;
  2. 提交一份"如何在 `python/sglang/srt/models/` 中新增模型"的分析文档。
