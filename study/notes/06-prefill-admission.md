# prefill 选批：`_get_new_batch_prefill_raw` 六步

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/scheduler.py` | `:3267` 主体（行号区间见下表）· `:3186` 外层可暂缓 prefill · `:3359-3361` 先续跑已有 chunk |
| `managers/schedule_policy.py` | `:511` `PrefillAdder` · `:537-545` batch 共用预算 · `:893-903` 扣减本轮预算 · `:1004-1020` 续跑与 hybrid SWA 停放 |

**六步对应的行号区间**（`scheduler.py`，对着代码读时按这个分段）

| 步 | 区间 | 内部关键行 |
|---|---|---|
| ① 快速门禁 | `3272-3322` | `:3285-3289` 空队列/满批 · `:3290` delayer · `:3307-3313` req slot · `:3235-3238` `get_num_allocatable_reqs` |
| ② 排序 | `3315-3316` | `policy.calc_priority()` 原地改 `waiting_queue` |
| ③ 建账 | `3324-3357` | `:3324-3339` 算 chunk size / tile · `:3340-3357` 构造 `PrefillAdder` |
| ④ 扫描准入 | `3376-3466` | `:3380-3465` 主循环 · `:3428` `add_one_req` · `:3450` 身份比较 · `:3462` break |
| ⑤ 提交 | `3468-3487` | `:3468-3471` 出队 · `:3473-3474` 被抢占者回队 · `:3487` 打时间戳 |
| ⑥ 物化 | `3489-3551` | `:3489-3499` `init_new` · `:3511` `prepare_for_extend` · `:3517-3528` 统计 · `:3538-3547` mixed chunk · `:3551` return |

---
`scheduler.py:3267`。一句话：**从 `waiting_queue` 中选出当前资源允许执行的请求，提交队列状态，
并物化一个 extend batch。**

```text
① 快速门禁
  ↓
② 按 policy 排序 waiting_queue
  ↓
③ 创建 PrefillAdder 预算账本
  ↓
④ prefix match 后逐个尝试准入
  ↓
⑤ 选中请求出队，更新 chunk/preemption 状态
  ↓
⑥ ScheduleBatch.init_new() + prepare_for_extend()
```

## ① 快速门禁

没有已有 `chunked_req` 时，以下快速门禁可返回 `(None, running_batch)`：`waiting_queue` 为空、
`running_batch.batch_is_full`、最少空闲 slot delayer 要求暂缓、没有 request slot。
优先级抢占和 hybrid SWA 的复位逻辑会影响 `batch_is_full`；测试钩子 `TEST_RETRACT` 也可直接返回。

可分配请求数：

```text
num_allocatable = min(pp_max_micro_batch_size - running_bs,  free_request_pool_slots)
```

这里只检查 **request slot 数量**；KV token 容量由 `PrefillAdder` 检查。

**已有 `chunked_req` 绕过上述空队列/满批、最少空闲 slot 延迟和 request-slot 快速检查**
（3285-3313），以便 PP 跨 microbatch 时仍能继续管理这个半截请求；`TEST_RETRACT` 检查没有该豁免。
进入本函数之前，外层 `_should_defer_prefill()` 仍可暂缓整次 prefill 选批（3186）。
进入 `add_chunked_req` 后，hybrid SWA 也可因 `_rem_tokens <= 0` 停放原请求（`schedule_policy.py:1017-1019`）。
所以“优先尝试续跑”不等于“每个调度轮都无条件执行”。

## ② 排序 ≠ 准入

`self.policy.calc_priority(waiting_queue, running_batch)` 原地调整候选顺序（LPM / FCFS / priority）。

- **Policy 回答"先尝试谁"。**
- **`PrefillAdder` 回答"谁真正放得下"。**

不要把 LPM/FCFS 理解成已经生成了 batch。

## ③ 建账

每次选批**临时创建**一本 `PrefillAdder`，初始状态来自：KV allocator 可用容量、tree cache 可驱逐容量、
已有 `running_batch` 的未来 decode 占用估算、`max_prefill_tokens` / chunk size / page size，
以及 SWA、Mamba、dLLM、tile budget 等专项约束。

`rem_chunk_tokens` 是**本轮整个 prefill batch 共用的 token 配额**，不是每个请求各得一份。
已有 chunk 先消耗配额（3359-3361），随后才扫描等待队列；每次 `_update_prefill_budget` 按
页对齐后的 extend 长度继续扣减。mixed chunk 模式构造账本时还先扣掉计划混入的 decode token 数。
启用动态 chunking 时，本轮初始配额可能由 `predict_next_chunk_size` 调整（3324-3330）。

## ④ 逐个准入

```text
检查 LoRA / req slot / HiCache 等前置条件
          ↓
req.init_next_round_input(tree_cache)     ← 算最新 prefix cache 命中
          ↓
adder.add_one_req(req)
          ↓
CONTINUE / NO_TOKEN / OTHER
```

**必须先做 prefix match，再做预算。**真实 extend 长度是：

```text
extend_len = len(full_fill_ids) - len(matched_prefix_indices)
```

同样是 1000-token prompt，命中 800 时只需为约 200 个新 token 做 extend；完全未命中则约 1000 个。
**同一个请求在不同时刻"能否加入"的结论可能完全不同——因为缓存状态变了。**

| 结果 | 含义 | 调度动作 |
|---|---|---|
| `CONTINUE` | 仍可继续尝试 | 扫描下一个 |
| `NO_TOKEN` | KV/状态池容量不足 | 标记容量压力并 break |
| `OTHER` | 计算量、chunk、请求数或 delay 限制到达 | break，但**不等同于持久内存已满** |

## ⑤ 提交

`adder.can_run_list` 为空 → 返回 `None`，所有等待请求留待重试。非空时：选中请求从 `waiting_queue` 移除、
被抢占请求重新入队、新的半截长请求写入 `self.chunked_req`、记录 forward entry time。

## ⑥ 物化

```python
new_batch = ScheduleBatch.init_new(can_run_list, ...)
new_batch.prepare_for_extend()
```

边界必须记清：

- `PrefillAdder` 是**预算预测和请求选择**；
- `ScheduleBatch.init_new()` 建立 batch 上下文；
- **`prepare_for_extend()` 才真正设置 `ForwardMode.EXTEND`、分配 req/KV slots、构造 forward 张量。**

普通模式返回 `(new prefill batch, 原 running decode batch)`；mixed chunk 把旧 decode 请求混入新 extend batch，
并返回一个**空的**独立 `running_batch`，避免同一请求被两个 batch 重复持有。

## 四种典型返回

| 场景 | `batch_to_run` | `running_batch` |
|---|---|---|
| 没等待请求 / 被 delay / 无 slot / 无人通过准入 | `None` | 原批，可能更新了 `batch_is_full` |
| 普通 prefill 成功 | 新 extend batch | 原批 |
| 优先级抢占后 prefill | 新 extend batch | 移除被抢占请求；被抢占者已回等待队列 |
| mixed chunk | prefill+decode 混合批 | 新建的空批 |

---

---

← [四个正确性级别的机制](05-batch-invariants.md)　|　[准入的核心：`add_one_req` 九道闸](07-add-one-req.md) →
