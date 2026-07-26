# 实验 2:离线 Engine API —— 不起服务器直接推理(约 1 小时)

## 目标

使用 `sglang.Engine` 在同一进程内做批量离线推理,不需要 HTTP 服务器。

## 步骤

### 1. 批量离线推理 100 条 prompt

```bash
python batch_inference_100.py --model-path Qwen/Qwen2.5-0.5B-Instruct
```

脚本会生成 100 条 prompt,批量提交给 Engine,并统计总耗时与吞吐(requests/s、tokens/s)。
Engine 内部的调度器会自动做连续批处理,大批量提交也不会 OOM。

参考官方示例:[examples/runtime/engine/offline_batch_inference.py](../../runtime/engine/offline_batch_inference.py)

### 2. async 与流式生成

参考 [examples/runtime/engine/offline_batch_inference_async.py](../../runtime/engine/offline_batch_inference_async.py),
将脚本改为 `async_generate`,体验流式逐 token 输出。

### 3. 调整采样参数

修改脚本中的 `sampling_params`,观察输出变化(参数含义见 [docs/basic_usage/sampling_params.md](../../../docs/basic_usage/sampling_params.md)):

- `temperature`:0(贪心、确定性)vs 1.0(随机性高)
- `top_p`:核采样阈值
- `max_new_tokens`:生成长度上限
- `stop`:停止字符串,如 `["\n"]`

## 产出

- 批量处理 100 条 prompt 的运行日志及吞吐统计(脚本会自动打印)。

## 思考题

1. 一次性提交 100 条 prompt 时,Engine 是"逐条串行"还是"动态组批"处理?从哪里可以观察到?
2. `temperature=0` 时两次运行的输出为什么完全一致?
