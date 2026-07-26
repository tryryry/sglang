# 实验 1:第一次跑起来 —— 启动服务器并发请求(约 1 小时)

## 目标

启动 SGLang 服务器,理解其多进程架构,并用三种方式发送请求。

## 步骤

### 1. 启动服务器

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000
```

### 2. 观察启动日志

在日志中找到并记录:

- 各子进程的启动信息:`TokenizerManager`(主进程)、`Scheduler`(GPU 调度子进程)、`DetokenizerManager`(反分词子进程)
- KV cache 池大小(`KV size` / `#token` 相关日志)
- CUDA Graph 捕获过程(`Capture cuda graph ...`)

### 3. 三种方式发请求

运行本目录下的脚本(需要先 `pip install openai requests`):

```bash
python send_requests.py --port 30000
```

该脚本依次演示:

1. **原生 API**:`POST /generate`(等价于 `curl http://localhost:30000/generate`)
2. **OpenAI 兼容 API**:用 `openai` SDK 调 `/v1/chat/completions`
3. **流式 vs 非流式**:对比 `stream=True` 与 `stream=False` 的收包行为

也可以直接用 curl 试原生 API:

```bash
curl -s http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}'
```

### 4. 管理接口

```bash
curl -s http://localhost:30000/health
curl -s http://localhost:30000/get_server_info | python -m json.tool | head -50
```

## 产出

- 三种请求方式的输出对比记录;
- 启动日志中标注出的三个子进程与 KV 池大小。

## 思考题

服务运行日志中每一行 decode 统计里的这些字段是什么含义?

- `#running-req`:当前正在 decode 的请求数;
- `#queue-req`:还在等待队列中未被调度的请求数;
- `token usage`:KV cache 池的占用比例。

请结合 `python/sglang/srt/managers/scheduler.py` 的日志打印代码验证你的理解。
