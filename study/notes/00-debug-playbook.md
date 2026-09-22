# Debug 速查：启动、请求、断点

> SGLang 学习笔记 · [返回总索引](../README.md)

可直接复制的调试命令，**调试时不用每次重新构造**。

- 下面的 `MODEL`、端口、venv 路径按自己的环境改一次即可。
- 端点和 CLI 参数都对着当前代码核实过（`entrypoints/http_server.py`、`entrypoints/v1_loads.py`、`server_args.py`）。

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `entrypoints/http_server.py` | 路由注册 · `:831` `/get_load`（已废弃的 shim） |
| `entrypoints/v1_loads.py` | `:97` `/v1/loads` 当前的负载端点 |
| `server_args.py` | 所有 CLI 参数 |

---

## 1. 启动

```bash
export MODEL=/root/models/Qwen3.5-2B
export PORT=30000
export CU=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13
```

### 1.1 调试配置（关掉会影响断点的一切）

`--disable-overlap-schedule` 让结果当轮落地，时序简单；两个 `--disable-*-cuda-graph`
让前向走 eager，能单步进 kernel 调用点。

```bash
PATH="$PWD/.venv/bin:$CU/bin:$PATH" CUDA_HOME="$CU" \
./.venv/bin/sglang serve \
  --model-path $MODEL --host 0.0.0.0 --port $PORT \
  --attention-backend triton --sampling-backend pytorch \
  --disable-overlap-schedule \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph
```

### 1.2 性能配置（观察真实调度行为时用）

去掉上面三个 `--disable-*`，overlap 和 CUDA graph 都开着——这才是
[四个正确性级别的机制](./05-batch-invariants.md) 里那些 overlap 时序问题会出现的模式。

### 1.3 VSCode launch.json

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug SGLang",
      "type": "debugpy",
      "request": "launch",
      "module": "sglang.launch_server",
      "python": "${workspaceFolder}/.venv/bin/python",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "subProcess": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/python",
        "CUDA_HOME": "${workspaceFolder}/.venv/lib/python3.12/site-packages/nvidia/cu13",
        "PATH": "${workspaceFolder}/.venv/bin:${workspaceFolder}/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:${env:PATH}"
      },
      "args": [
        "--model-path", "/root/models/Qwen3.5-2B",
        "--host", "0.0.0.0", "--port", "30000",
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
        "--disable-overlap-schedule",
        "--disable-prefill-cuda-graph",
        "--disable-decode-cuda-graph"
      ]
    }
  ]
}
```

`subProcess: true` 是必须的——Scheduler 是 `mp.Process` fork 出来的独立进程，
不开这个断点打不进调度代码。

---

## 2. 基础请求

### 2.1 非流式

```bash
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

### 2.2 流式

```bash
curl -N http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

### 2.3 长 prompt（触发 chunked prefill）

```bash
LONG=$(python3 -c "print('请复述下面这段话。' + '这是一段用来填充上下文的测试文本。' * 2000)")
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "$MODEL" --arg c "$LONG" \
        '{model:$m, messages:[{role:"user",content:$c}], max_tokens:16, temperature:0}')"
```

配合 `--chunked-prefill-size 512` 启动，可以稳定走到
[Chunked prefill](./09-chunked-prefill.md) 的切片路径。

---

## 3. 多模态

### 3.1 网络图

```bash
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
      {"type": "text", "text": "请描述这张图片。"}
    ]}],
    "max_tokens": 16, "temperature": 0
  }'
```

### 3.2 本地 base64（离线可用，32×32 的纯色小图）

不依赖外网，适合在没有出网的机器上调多模态路径。

```bash
IMG='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=='
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "$MODEL" --arg u "$IMG" \
        '{model:$m, messages:[{role:"user",content:[
            {type:"image_url", image_url:{url:$u}},
            {type:"text", text:"Describe the image."}]}],
          max_tokens:8, temperature:0}')"
```

### 3.3 双图（测多图 token 计费）

把 §3.1 的 `content` 数组里放两个 `image_url` 再加一个 `text` 即可。

---

## 4. 看内部状态

### 4.1 这次请求命中了多少前缀

`return_meta_info: true` 让响应带上 `meta_info`，`cached_tokens` 就是
[存量 vs 流量](./08-token-budget.md) 里 `prefix_indices` 的长度。

```bash
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 8, "temperature": 0, "return_meta_info": true
  }' | jq '.choices[0].meta_info | {prompt_tokens, cached_tokens, image_tokens}'
```

**用法**：同一个请求连发两次，第二次的 `cached_tokens` 应该接近 `prompt_tokens`
（差 1，因为前缀匹配上限是 `input_len - 1`）。

### 4.2 调度器负载

```bash
curl -s http://127.0.0.1:$PORT/v1/loads | jq          # 当前端点
curl -s http://127.0.0.1:$PORT/get_load  | jq         # 旧 shim，字段更扁平
```

`num_running_reqs` 只增不减 → 多半是
[prefill-only 请求没被清理](./05-batch-invariants.md)。

### 4.3 清空 radix cache（复现冷启动）

```bash
curl -s -X POST http://127.0.0.1:$PORT/flush_cache
```

清完再发同一个请求，`cached_tokens` 会掉回 0——可以用来对比
[两本 token 账](./08-token-budget.md) 里命中/未命中两种情形的预算。

### 4.4 其它

```bash
curl -s http://127.0.0.1:$PORT/health                 # 存活
curl -s http://127.0.0.1:$PORT/health_generate        # 能真的跑通一次生成
curl -s http://127.0.0.1:$PORT/get_server_info | jq   # 生效的 server_args
curl -s http://127.0.0.1:$PORT/get_model_info  | jq
curl -s -X POST http://127.0.0.1:$PORT/abort_request \
  -H "Content-Type: application/json" -d '{"rid": "<rid>"}'      # 单个
curl -s -X POST http://127.0.0.1:$PORT/abort_request \
  -H "Content-Type: application/json" -d '{"abort_all": true}'   # 全部
```

### 4.5 抓 profile

```bash
curl -s -X POST http://127.0.0.1:$PORT/start_profile
#   ... 期间发几个请求 ...
curl -s -X POST http://127.0.0.1:$PORT/stop_profile
```

---

## 5. 断点清单

按笔记分组，打在这些位置就能顺着主线走一遍。

### 请求入口 → [请求的一生](./01-request-lifecycle.md)

| 位置 | 看什么 |
|---|---|
| `entrypoints/openai/serving_chat.py:1100` | 原始 `messages` |
| `:1386` | 渲染 chat template |
| `:1394` | 得到 `prompt_ids` |
| `:966` | 构造内部请求 |
| `managers/tokenizer_manager.py:768` | 请求标准化 |
| `:959` | 确认走 `input_ids` 直通分支 |
| `:1334` | 生成最终 `TokenizedGenerateReqInput` |

### 调度主循环 → [`get_next_batch_to_run` 主干](./04-scheduling-loop.md)

| 位置 | 看什么 |
|---|---|
| `managers/scheduler.py:3095` | 进入一轮调度 |
| `:3162` | `last_batch.filter_batch()`，看谁被过滤 |
| `:3172` | merge 进 `running_batch`，看"毕业" |
| `:3208` / `:3212` | prefill 还是 decode 这一轮 |

### 准入 → [六步](./06-prefill-admission.md) / [九道闸](./07-add-one-req.md) / [两本账](./08-token-budget.md)

| 位置 | 看什么 |
|---|---|
| `managers/scheduler.py:3267` | 进入选批 |
| `:3340` | 建 `PrefillAdder`，看四本预算的初值 |
| `:3428` | 每个请求的 `add_one_req` 调用 |
| `:3450` | `added = ... req is adder.can_run_list[-1]`，验证返回值 ≠ 是否加入 |
| `managers/schedule_policy.py:1243` | 闸 2，看 `total_tokens` vs `rem_total_tokens` |
| `:1332` | 锁内重算 `input_tokens` |
| `:864` | `_update_prefill_budget`，看余额怎么被扣 |

---

## 6. 故意触发特定路径

| 想看什么 | 怎么做 |
|---|---|
| **chunked prefill 切片** | `--chunked-prefill-size 512` + §2.3 的长 prompt |
| **关掉 chunk，看队首阻塞** | `--chunked-prefill-size -1` + 长 prompt + 并发短请求 |
| **retract / KV 打满** | 调小 `--mem-fraction-static`（如 `0.5`）+ 高并发长输出，看日志 `KV cache pool is full. Retract requests.` |
| **每次都全量 prefill** | `--disable-radix-cache`，`cached_tokens` 恒为 0 |
| **`max_prefill_tokens` 闸生效** | `--chunked-prefill-size -1 --max-prefill-tokens 2048`，只有这时闸 5 才不被跳过 |
| **prefill 让路给 decode** | `--prefill-decode-interval 4`，看 prefill 被推迟 |
| **改调度顺序** | `--schedule-policy fcfs` / `lpm`，对比 `waiting_queue` 的尝试顺序 |
| **限制并发** | `--max-running-requests 4`，容易观察到 `batch_is_full` |

---

## 7. 日志里值得 grep 的串

| 字符串 | 出处 | 含义 | 相关笔记 |
|---|---|---|---|
| `KV cache pool is full. Retract requests.` | `scheduler.py:3635` | KV 打满触发抢占回退，**容量告警** | [04](./04-scheduling-loop.md) |
| `Prefill batch, ...` | `metrics_reporter.py:611` | 每轮 prefill 批的组成 | [06](./06-prefill-admission.md) |
| `Decode batch, ...` | `metrics_reporter.py:816` | 每轮 decode 批的规模 | [04](./04-scheduling-loop.md) |

`Prefill batch` 那行的字段（`metrics_reporter.py:611-618`）：

```text
Prefill batch, #new-seq: N, #new-token: N, #cached-token: N, <token usage>,
               #running-req: N, #queue-req: N, #pending-token: N
                            ↑              ↑
            = adder.log_input_tokens  = adder.log_hit_tokens
              这一轮真正要算的        这一轮靠前缀命中省下的
```

`#new-token` / `#cached-token` 正是 [两本 token 账](./08-token-budget.md) 里
`rem_input_tokens` 扣减量和 `prefix_len` 的累计值——**线上看这两个数的比值就知道前缀复用效果。**
`#queue-req` 持续不降而 `#running-req` 不涨，多半是 `batch_is_full` 没复位（[04](./04-scheduling-loop.md)）。

---

[请求的一生](01-request-lifecycle.md) →
