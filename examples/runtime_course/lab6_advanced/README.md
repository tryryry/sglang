# 实验 6(进阶,选做):高级特性体验(约 2 小时)

任选其二完成。

## 1. 结构化输出(JSON Schema 约束)

启动服务器后运行:

```bash
python structured_output.py --port 30000
```

脚本用 `json_schema` 约束模型输出,保证生成的一定是合法且符合 schema 的 JSON。
背后由语法引擎(xgrammar / llguidance / outlines)在每步采样时对 logits 做 mask,
代码见 `python/sglang/srt/constrained/`。

**观察点**:对比同一 prompt 有/无 `json_schema` 约束的输出;故意让 schema 复杂化,看约束是否仍严格生效。

## 2. 投机解码(Speculative Decoding)

参考 [examples/runtime/engine/offline_batch_inference_eagle.py](../../runtime/engine/offline_batch_inference_eagle.py),
用 EAGLE draft 模型启动,并与不开投机解码对比 decode 吞吐(日志中 `gen throughput` 或 bench_serving 的 ITL):

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <eagle-draft-model> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16
```

相关代码在 `python/sglang/srt/speculative/`。

**观察点**:日志中的 accept length(平均每步接受多少 draft token);temperature 调高后接受率如何变化。

## 3. 量化(FP8)

加载 FP8 模型,对比显存占用(`nvidia-smi`)与吞吐:

```bash
python -m sglang.launch_server --model-path neuralmagic/Qwen2.5-0.5B-Instruct-FP8 --port 30000
# 或对 BF16 权重做在线量化:
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --quantization fp8 --port 30000
```

量化实现见 `python/sglang/srt/layers/quantization/`。

## 4. 性能剖析(Torch Profiler)

参考 [examples/profiler/](../../profiler/),生成 Chrome trace:

```bash
# 服务器端以环境变量指定 trace 输出目录后,通过 /start_profile 与 /stop_profile 控制采样
curl -s http://localhost:30000/start_profile -H "Content-Type: application/json" \
    -d '{"output_dir": "/tmp/sglang_trace", "num_steps": 10}'
# ... 发送一些请求 ...
curl -s http://localhost:30000/stop_profile
```

将生成的 `.trace.json.gz` 拖入 [Perfetto](https://ui.perfetto.dev/) 或 `chrome://tracing` 查看 kernel 时间线。

**观察点**:decode 一步中 attention kernel、GEMM、采样各占多少时间;CUDA Graph 重放与非 graph 模式的 launch 间隙差异。

## 5. 张量并行(多卡)

多卡环境下:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --tp 2 --port 30000
```

**观察点**:`nvidia-smi` 中两张卡的显存/利用率分布;启动日志中每个 TP rank 的 Scheduler 子进程。
