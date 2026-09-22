# `get_next_batch_to_run` 主干

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/scheduler.py` | `:3095` 主实现 · `:1778/1822` 事件循环 · `:1216` `_should_defer_prefill` · `:3581` `update_running_batch` · `:3635` retract 日志 |
| | 前置清理三件套：`:3008` `process_pending_chunked_abort` · `:2896` `_abort_on_waiting_timeout` · `:1672` `_abort_on_running_timeout` |
| `managers/scheduler_components/dp_attn.py` | `:425` `maybe_prepare_mlp_sync_batch`，DP-attn 下各 rank 同步 forward mode |
| `disaggregation/prefill.py` | `:546` PD prefill 侧 · `:552` `batch_is_full` HACK |
| `disaggregation/decode.py` | `:2555` PD decode 侧 |

---
`scheduler.py:3095`。四段：**清理 → 毕业 → 选 prefill → 退化到 decode**。

```text
① 前置清理 (3098-3105)
   process_pending_chunked_abort()   延迟到安全点再 abort
   _abort_on_waiting_timeout()       排队超时踢掉（回 503）
   _abort_on_running_timeout()       运行超时只打 to_finish，由后面 filter 统一收割
          ↓
② last_batch(extend) 合入 running_batch (3107-3182)
   chunked_req_to_exclude = {还没 prefill 完的 req}
   last_batch.filter_batch(exclude)  踢掉 finished + 黑名单
   running_batch.merge_batch(last_batch)
          ↓
③ 尝试组一个新的 prefill batch (3184-3191)
   new_batch = get_new_batch_prefill(running_batch)
          ↓
④ 二选一 (3206-3215)
   new_batch 非空 → 跑 prefill（优先！）
   否则          → update_running_batch() 跑 decode
   都没有        → None（空转 on_idle）
          ↓
尾处理：DP-attn sync / ngram / 打时间戳
return NextBatchPlan(ret, running_batch)
```

## prefill 优先是硬编码的

```python
if new_batch is not None:
    ret = new_batch                      # ★ 有 prefill 就先跑 prefill
else:
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = self.update_running_batch(running_batch)
        ret = running_batch if not running_batch.is_empty() else None
```

吞吐友好、对 ITL 不友好，所以才有一堆反向节流阀：`--prefill-decode-interval`、`prefill_delayer`、
`min_free_slots_delayer`。**它们全都作用在第 ③ 步之前（提前返回 `None`），而不是在第 ④ 步改优先级。**

`_should_defer_prefill()` / `_arm_prefill_decode_interval()`（`scheduler.py:1216/1223`）实现
`--prefill-decode-interval`：跑一次 prefill 后强制接下来 N 步只做 decode。`_arm` 里特意用
`batch.is_extend_in_batch`（全局视角）而不是本地 `forward_mode`，保证 DP-attn 下各 rank 节奏一致。

## `batch_is_full` 是必须手动复位的刹车

它是 prefill adder 设置的"别再塞新请求了"标志，**跨轮持久**。一旦有 req 离开批就必须解除，
否则高并发下会永远不再接新请求——形成"有资源但不收人"的停滞。

代码里至少有 4 处补偿 + 1 处 HACK 注释：

| 位置 | 场景 |
|---|---|
| `scheduler.py:3164` | extend batch 过滤掉 chunked/finished 后批变小 |
| `scheduler.py:3182` | prefill-only 批清空 |
| `scheduler.py:3653` | `update_running_batch` 里批变小 |
| `scheduler.py:3145` | hisparse 分支 |
| `prefill.py:552` | disagg prefill 每轮开头**无条件** `= False`，注释直说 "otherwise it hangs under high concurrency" |

**这是本函数里出现频率最高的一类 bug。**

## KV 不足的兜底是 retract 不是 OOM

`update_running_batch`（`scheduler.py:3581`）做三件事：

1. `filter_batch()` 收割 finished；
2. `check_decode_mem()` 不过 → `retract_decode()` **抢占回退**：把部分请求踢回 `waiting_queue`，
   腾 KV，同时**上调 `new_token_ratio`**（更保守地预留），打 warning `KV cache pool is full. Retract requests.`；
   没触发则 `new_token_ratio_tracker.decay_step()` 慢慢放开；
3. `prepare_for_decode()` 准备张量。

线上看到 `Retract requests` 日志 = KV 池打满，应关注 `--mem-fraction-static` 和并发。
**频繁 retraction 会造成重复 prefill，应视为容量告警。**

## 三个变体的对比

| | 普通 | disagg PREFILL | disagg DECODE |
|---|---|---|---|
| 入口 | `scheduler.py:3095` | `prefill.py:546` | `decode.py:2555` |
| last_batch merge | 有 | 用 `process_prefill_chunk()` 代替 | 用 `get_new_prebuilt_batch()` 代替 |
| prefill | `get_new_batch_prefill` | 同上 | 无 |
| decode | `update_running_batch` | 无 | `update_running_batch` |

主实现 = 两个变体的并集，PD 分离只是把 prefill/decode 两半拆到不同进程。

---

---

← [调度器的五个状态变量](03-scheduler-state.md)　|　[四个正确性级别的机制](05-batch-invariants.md) →
