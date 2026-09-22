# KV 池为什么决定一切：从显存预留到并发

> SGLang 学习笔记 · [返回总索引](../README.md)

这条链横跨 [chunked prefill](./09-chunked-prefill.md)（预留多大）、
[两本 token 账](./08-token-budget.md)（怎么花）、
[调度主干](./04-scheduling-loop.md)（花超了怎么办），单独拎出来讲一次。

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `server_args.py` | `:5309` 仅当 `mem_fraction_static is None` 才推导 · `:5311-5314` post-capture 分支 · `:5425-5432` `pre_capture_activation_reserve_mb` · `:5336-5343` 推出 `mem_fraction_static` |
| `mem_cache/kv_cache_configurator.py` | `:1827` `_profile_available_bytes` · `:1838-1852` `slack_gb` / `rest_memory` · `:1999-2002` 字节→token→并发 · `:1936` `resolve_max_num_reqs` |
| `mem_cache/multi_ended_allocator.py` | `:383` `available_size()` 单位是 TOKEN |
| `managers/schedule_policy.py` | `:671` `rem_total_tokens` 运行时余量 |
| `managers/scheduler.py` | `:3581` `update_running_batch` · `:3635` retract 日志 |

---

## 1. 完整链条

```text
chunked_prefill_size ↓                      （或任何调小 activation 预留的做法）
  → activation_tokens ↓                     server_args.py:5427
  → reserved_mem ↓                          server_args.py:5429
  → mem_fraction_static ↑                   server_args.py:5339
  → slack_gb ↓                              kv_cache_configurator.py:1838
  → rest_memory ↑ → available_bytes ↑       :1852 / :1874
  → max_total_num_tokens ↑                  :2000
  → max_running_requests ↑                  :2001
  → rem_total_tokens ↑                      schedule_policy.py:671
```

三个关键等式：

```python
mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)        # :5339
slack_gb            = pre_model_load_memory * (1 - mem_fraction_static)   # :1838
rest_memory         = available_gpu_memory - slack_gb - mm_reservation_gb # :1852
```

`mem_fraction_static` 字面上就是 `1 − reserved_mem/gpu_mem`。

### ⚠️ 两个前提，缺一条链就断

| 前提 | 位置 | 断链后果 |
|---|---|---|
| **没有显式传 `--mem-fraction-static`** | `:5309` 的 `if self.mem_fraction_static is None:` | 传了就用你给的值，`chunked_prefill_size` 不再影响 KV 池大小 |
| **不是 post-capture sizing** | `:5311-5314`，`post_capture_kv_sizing_planned()` 为真时只留 `1536 + tp*pp/8*1024` 的地板 | 它在 CUDA graph 捕获后重新量空闲显存，不需要提前猜，activation 预留被跳过 |

**所以"调小 chunk 能换来 KV 池"只在默认、自动推导的配置下成立。**

---

## 2. 静态上限 ≠ 实际并发

链条末端有两个不同的东西，别混：

| | 是什么 | 怎么来 |
|---|---|---|
| `max_running_requests` | **静态粗上限**，启动时定死 | `resolve_max_num_reqs(max_total_num_tokens)`，`:1936` |
| `rem_total_tokens` | **每轮实时余量**，真正决定这一刻能收几个 | `available_size() + evictable_size() − offset`，`schedule_policy.py:671` |

静态上限的公式（`:1940-1949`）：

```python
estimated = int(token_capacity / self.model_config.context_len * 512)
estimated = max(min(estimated, 4096), 2048)          # 夹在 [2048, 4096]
max_num_reqs = min(estimated, token_capacity // 2)   # 用户没指定时
# 用户指定了也要被压：min(requested_per_worker, token_capacity // 2)
```

注意它被夹在 `[2048, 4096]`，**绝大多数部署里根本不是瓶颈**。真正卡住并发的是运行时的
`rem_total_tokens`——[闸 2](./07-add-one-req.md) 拿它跟每个请求的 `total_tokens` 比。

> 结论：**"并发"在 SGLang 里实质是"KV token 预算"**，不是一个请求计数。
> 一个 100 并发 × 2k 上下文的负载，和 10 并发 × 20k 上下文，对池子的压力是一样的。

---

## 3. 为什么并发值钱：decode 是 memory-bound 的

这是整条链的**终点意义**。

decode 每一步要把**整个模型权重**从 HBM 读一遍，才能为每条序列产出 1 个 token。
读权重的代价**与批大小无关**——B=1 读一遍产出 1 个 token，B=64 读同样一遍产出 64 个。

```text
每步读取字节 ≈ W  +  B × S × kv_per_token
                ↑        ↑
            权重，与 B 无关   KV，随 B 线性增长

每 token 成本 = W/B + S × kv_per_token
                 ↑
            B 越大，这项越小 → 这就是并发的全部价值
```

**一个示意性估算**（7B fp16 权重 ≈ 14 GB，H100 HBM ≈ 3.35 TB/s，序列长 2k，56 KB/token）：

| 批大小 B | 读权重 | 读 KV | 每步耗时 | 吞吐 | 每 token 延迟 |
|---|---|---|---|---|---|
| 1 | 4.2 ms | ~0.1 ms | ≈ 4.3 ms | ≈ 230 tok/s | 4.3 ms |
| 64 | 4.2 ms | 2.2 ms | ≈ 6.4 ms | ≈ 10,000 tok/s | 6.4 ms |
| 256 | 4.2 ms | 8.7 ms | ≈ 12.9 ms | ≈ 20,000 tok/s | 12.9 ms |

（数量级示意，不是实测。）读出三件事：

1. **B=1 → 64：吞吐涨 40 倍，延迟只涨 50%。**这段几乎是白捡的——GPU 本来就在空转着搬权重。
2. **B=64 → 256：吞吐只翻倍，延迟也翻倍。**越过权重主导区之后，收益递减。
3. 并发低的时候，**带宽绝大部分花在反复搬运权重上**，产出很少 token。

SGLang 的默认值印证了这个目标区间——decode CUDA graph 的 `max_bs` 按显存分档
（`server_args.py:5170-5220`）：

| GPU | `decode max_bs` 默认 |
|---|---|
| T4 | 8 |
| A10 / 4090 | 24（tp<4）/ 80 |
| A100-40G / L40 | 32 / 160 |
| H100 / H200 | 256 / 512 |
| B200 / MI300 | 512 |

**大卡把 decode batch 捕获到 512，就是因为那才是它的效率区间。**

---

## 4. 并发的代价与退化

### 代价：ITL

上表第三列已经说明——B 越大，每步越慢，**单个用户看到的 inter-token latency 越高**。
吞吐和延迟在这里是直接对立的，这也是
[chunked prefill 那边 ITL 讨论](./09-chunked-prefill.md) 的同一个张力。

### 退化：池子不够时不是"慢一点"，是雪崩

KV 池偏小会踩进两条退化路径，两条都在笔记里有：

| 症状 | 机制 | 出处 |
|---|---|---|
| 日志刷 `KV cache pool is full. Retract requests.` | `check_decode_mem()` 不过 → `retract_decode()` 把 running 请求踢回队列 → **重新 prefill 一遍** | [调度主干](./04-scheduling-loop.md)，`scheduler.py:3635` |
| `#queue-req` 持续不降，`#running-req` 不涨 | 闸 2 返回 `NO_TOKEN` → `batch_is_full = True` → 主循环 `break`，队首堵死 | [九道闸](./07-add-one-req.md)、[Chunked prefill §2](./09-chunked-prefill.md) |

**retract 是负反馈里最坏的一种**：它把已经算完的 prefill 作废，腾出的 KV 又被新请求占走，
容易形成反复 prefill 的震荡。所以看到 `Retract requests` 不是"有点紧"，是**容量配置需要改**。

---

## 5. 二阶好处：KV 池同时是 radix cache 的家

```python
rem_total_tokens = allocator.available_size() + tree_cache.evictable_size() - offset
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                               可驱逐的部分，就是前缀缓存
```

池子里没被 running 请求占住的部分，radix cache 拿去存前缀。**池子越大，能留住的前缀越多，
命中率越高，需要真正计算的 prefill 越少**（见 [两本 token 账](./08-token-budget.md) 的
device 命中一行）。

这是个正反馈：更大的池子 → 更高命中率 → 每个请求的 `cand_extend_input_len` 更小 →
占用更少 → 能收更多请求。

用 [Debug 速查](./00-debug-playbook.md) §4.1 的 `cached_tokens` 和日志里的
`#cached-token / #new-token` 比值可以直接观察到这个效果。

---

## 一句话总结

> KV 池大小是 SGLang 最重要的一个容量参数：它同时决定**并发上限**、**前缀缓存容量**
> 和**是否会触发 retract 雪崩**。而并发之所以值钱，是因为 decode 受 HBM 带宽约束——
> 权重每步都要读一遍，批越大这笔固定成本被摊得越薄。
>
> 但它买的是**吞吐**，付的是 **ITL**。两头都要的话，得靠
> `--prefill-decode-interval` / `--enable-mixed-chunk` 这类调度手段去平衡，
> 而不是单纯把池子调大。

---

← [Chunked prefill](09-chunked-prefill.md)　|　[最容易踩的坑](99-pitfalls.md) →
