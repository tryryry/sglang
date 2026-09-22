# 四个正确性级别的机制

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `model_executor/forward_batch_info.py` | `:104` `ForwardMode` 九种模式 |
| `managers/schedule_batch.py` | `:3182` `filter_batch` · `:1225` `is_prefill_only` · `:1680` `update_finish_state` |
| `managers/scheduler.py` | `:3107-3182` 毕业段 · `:3131` 有新增 KV 才 stash · `:3211` 跳过 decode · `:3359` 优先尝试续跑已有 chunk |
| `managers/schedule_policy.py` | `:1004` 续跑 chunk；`:1017-1020` hybrid SWA 可停放，其他普通路径有预算回退 |
| `managers/tp_worker.py` | `:658` prefill-only 不采样 |
| `managers/scheduler_components/batch_result_processor.py` | `:316-320` final chunk 提交输出 token · `:327` `release_kv_cache` · `:360` 中间 chunk 只减 inflight 计数 · `:410` 假 token |

---
这四个不是优化，**少一个就出错**。

## 1. `forward_mode.is_extend()` 门卫 —— 防止 merge 自身

`ForwardMode`（`forward_batch_info.py:104`）九种模式中 `is_extend()` 为真的：
`EXTEND` / `MIXED` / `TARGET_VERIFY` / `SPLIT_PREFILL` / `DLLM_EXTEND`（`DRAFT_EXTEND_V2` 需显式打开）；
为假的：`DECODE` / `IDLE` / `PREBUILT`。

> SGLang 叫 extend 而不是 prefill，是因为前缀 KV 可能已在 radix cache 命中，实际只"延长"未命中的那段。

3147 处这个判断是"毕业"的门卫，等价语义是**"上一轮有没有产生一批需要转入 decode 稳态的新请求"**。

没有它会怎样——`merge_batch` 的写法是 `self.reqs = self.reqs + other.reqs`：

```text
prefill 轮:  running={C,D}  last={A,B}   不同对象 → merge 有意义 → {C,D,A,B}
decode  轮:  running={C,D}  last={C,D}   同一对象 → merge 是灾难 → {C,D,C,D}
                            ^^^^^^^^^^   is_extend() == False，门卫挡住
```

batch size 每步翻倍，`req_pool_indices` 同一 slot 出现两次，KV 写入冲突、采样重复出 token，**一步就炸**。

用 `forward_mode` 而不是 `last_batch is not running_batch` 判断，是因为 PP / overlap 下还有 IDLE 批、
`batch.copy()` 等情况，**语义判断比 identity 判断更稳**。

（overlap 下 `result_queue.append((batch.copy(), result))` 压的是 copy，但 `self.last_batch = batch`
存的仍是原对象，所以推理同样成立。）

## 2. `chunked_req_to_exclude` —— 防止半截请求毕业

一张**"这一轮不许毕业进 decode 的请求"黑名单**，唯一用途是喂给 `last_batch.filter_batch()`。

为什么需要：10 万 token 的长 prompt 受 `--chunked-prefill-size` 限制要分多轮算，
**只有最后一块的结果才提交为生成输出 token**；中间块即使执行了采样，也不会 append 到 `output_ids`。
但轮 1 跑完后这个请求确实出现在 `last_batch.reqs` 里，
且 `finished()` 是 False——按默认规则会被 merge 进 `running_batch`，可它**没有 output token 可 decode**，
进 decode 批直接状态错乱。`finished()` 判断不了它，所以需要额外黑名单。

名单里三类：

| 来源 | 行 | 说明 |
|---|---|---|
| dLLM 的 staging 请求 | 3110-3119 | |
| `self.chunked_req` | 3121-3132 | **scheduler 全局**的"我现在正在分块处理谁"；仅当 `extend_range.end > len(prefix_indices)` 时 stash 新增 KV，由所用 cache 实现推进前缀 |
| `last_batch.chunked_req` | 3152-3155 | **那个 batch 创建时拍下的快照**，PP 下可能已**过期** |

后两者的区别在 PP 下才显现：某 microbatch 手里的 `chunked_req` 可能已被别的 microbatch 喂完
（`self.chunked_req` 已置 None），这个快照就是过期的——过期的也得拦。用 `set` 正是因为三个来源会重叠。

**注意：拦下 ≠ 丢弃。** 尚未完成 prefill 的活动请求仍由 `self.chunked_req` 持有，进入下一次
prefill 选批时优先尝试续跑，并绕过空等待队列、`batch_is_full`、最少空闲 slot 延迟和 request-slot
快速门禁。**这不是无条件执行**：hybrid SWA 在 `add_chunked_req` 算得 `_rem_tokens <= 0` 时
直接返回原请求，本轮不加入 `can_run_list`；请求仍被追踪，等待后续容量恢复。
普通非 dLLM、非 hybrid-SWA 路径才会在该情况下回退到 `rem_chunk_tokens`。
外层 prefill 节奏也可能暂缓续跑，详见 [prefill 选批](./06-prefill-admission.md)。

因此不变量是**未完成 chunk 的所有权与 KV 状态不能丢失，且不能提前进入 decode**；
合法停放一轮本身不等于内存泄漏。PP 的过期 batch 快照也不意味着 `self.chunked_req` 此刻仍持有它。

```text
轮 k:   last_batch = {A(chunk中), B(prefill完), C(已finish)}
        ↓ filter_batch(exclude={A})
        ├─ A → 黑名单拦下，留在 self.chunked_req，下轮继续喂
        ├─ C → finished()，淘汰
        └─ B → 保留，merge 进 running_batch 开始 decode
```

## 3. `filter_batch()` —— 并行数组的收割器

**前提：`ScheduleBatch` 是"并行数组的结构体"**，不是简单的请求列表：

```text
索引                  0        1        2        3
reqs                [A,       B,       C,       D]
req_pool_indices    [17,      3,       42,      8]      ← GPU 上的 req slot 号
seq_lens            [128,     512,     64,      900]
sampling_info.temperatures  [...]                        ← 同样按位置对齐
```

GPU kernel 全靠**位置**索引。摘掉 C 不能只 `reqs.remove(C)`——必须同时删掉所有数组的第 2 个位置，
否则 `reqs[2]` 是 D 而 `seq_lens[2]` 还是 C 的 64，整批错位。

**为什么必须做**：`req.finished()` 由上一轮 batch result processor 打上，而**同一处已经把 KV 还回去了**
（`batch_result_processor.py:327` 的 `release_kv_cache`）。所以 C 在数组里还占着位置，
但 `req_pool_indices[2] = 42` 这个 slot **已经还给分配器、随时可能分给新请求**。不 filter 会：

- 白算一个死请求的 attention；
- 往 slot 42 写 KV —— 而 42 可能已属于别人，**直接踩内存**；
- 采样还给它出 token，然后试图发给已关闭的连接。

顺带三件事：作废派生值（`out_cache_loc`/`seq_lens_sum`/`mamba_*` 置 None）、重算聚合标志
（`return_logprob = any(...)`——批里最后一个要 logprob 的请求走了，整批就不用再算）、两条快速路径。

四个调用点：

| 行 | 场景 |
|---|---|
| 3162 | prefill 批毕业前，踢掉 finished + 黑名单（本篇） |
| 3180 | **prefill-only 批的专用清理**（本篇） |
| 3540 | mixed chunk：decode 请求拼进 extend 批之前先清一遍 |
| 3585 | `update_running_batch` 里，decode 主路径的常规收割 |

## 4. prefill-only 跳过 decode

`is_prefill_only` = `max_new_tokens == 0`（embedding、reward model、classify/score、只要 input logprob）。
批级标志是 `all(...)` 的结果。

**它们在 prefill 那一步就彻底 finished 了**，三步连锁：

1. tp_worker **根本不采样**（`tp_worker.py:658`），填 0 占位——embedding 模型甚至没有 lm_head，
   decode 对它们**在数学上就是未定义的**；
2. 结果处理时塞一个假 token `req.output_ids.append(0)`（`batch_result_processor.py:410`）；
3. `update_finish_state` 判 `len(output_ids) >= max_new_tokens` → `1 >= 0` 恒成立 → **立刻 finished**，
   紧接着 `release_kv_cache` 把 KV 和 req slot 还了回去。

所以对它们跑 decode 就是 本篇 那个踩内存场景。

**那它们怎么会进 `running_batch` 的？** overlap 时序：结果处理排在 `get_next_batch_to_run` **之后**。

```text
iter N:    prefill-only 批 P 发射。last_batch = P。结果还没落地。

iter N+1:  3162 filter_batch → P 的请求此刻 finished()=False（结果还没处理！）→ 一个没删
           3172 merge 进 running_batch          ★ 就这样进来了
           3179 is_prefill_only → filter_batch() → 仍没 finished，删不掉
           3211 跳过 decode                     ← 这道闸保证了正确性
           ... 本轮稍后 pop_and_process() 才落地 → 全部 finished + KV 释放

iter N+2:  3180 filter_batch() → 这次才真正清干净
```

这解释了两行代码为什么**配套存在**：3211 负责"不要跑"，3180 负责"清出去"。
而 3180 必须放在 `if last_batch ...` 块**外面**，是为了流量停了之后
（`last_batch` 变 None、再没新批进来）仍会执行，否则 `/v1/loads` 的 `num_running_reqs` 永久卡在非零值。

**混合批**（既有生成请求又有 embedding）不需要特殊处理：`is_prefill_only` 是 `all(...)` → False
→ 正常走 decode，那些 prefill-only 请求已 finished，被 3585 的常规收割干掉。

---

---

← [`get_next_batch_to_run` 主干](04-scheduling-loop.md)　|　[prefill 选批：`_get_new_batch_prefill_raw` 六步](06-prefill-admission.md) →
