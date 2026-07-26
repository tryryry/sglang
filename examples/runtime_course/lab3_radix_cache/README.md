# 实验 3:RadixAttention 前缀缓存实测(约 1.5 小时)

## 目标

亲手验证 RadixAttention 前缀缓存的效果:相同前缀的请求可以复用已计算的 KV cache,大幅降低首 token 延迟(TTFT)。

## 步骤

### 1. 启动服务器(开启 radix cache,默认即开启)

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000
```

### 2. 运行前缀缓存 benchmark

```bash
python prefix_cache_benchmark.py --port 30000
```

脚本会:

1. 用同一个约 1000 token 的长 system prompt 发第一次请求(冷启动,无缓存);
2. 立刻发第二次相同前缀的请求,对比返回的 `cached_tokens` 与 TTFT;
3. 调用 `POST /flush_cache` 清空缓存后重发,观察命中归零;
4. 模拟一个 5 轮多轮对话,统计每一轮的缓存命中率。

同时观察服务器日志中每次 prefill 的 `#cached-token` / `cache hit rate` 字段。

### 3. 关闭 radix cache 重跑

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000 --disable-radix-cache
```

再次运行:

```bash
python prefix_cache_benchmark.py --port 30000
```

此时第二次请求的 `cached_tokens` 应为 0,TTFT 与第一次接近。

## 产出

填写下表("有/无 radix cache 的 TTFT 对比"):

| 场景 | cached_tokens | TTFT (s) |
|---|---|---|
| radix cache 开启,第 1 次请求 | | |
| radix cache 开启,第 2 次请求 | | |
| flush_cache 后重发 | | |
| radix cache 关闭,第 2 次请求 | | |

## 思考题

1. 多轮对话中,第 N 轮的缓存命中 token 数大约等于什么?
2. radix tree 与简单的"前缀哈希表"相比,优势在哪里?(提示:部分前缀匹配)
3. KV cache 池满时,radix tree 中的缓存如何被驱逐?(阅读 `python/sglang/srt/mem_cache/` 下的 LRU 驱逐逻辑)
