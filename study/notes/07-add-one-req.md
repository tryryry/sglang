# 准入的核心：`add_one_req` 九道闸

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/schedule_policy.py` | `:1208` `add_one_req` · `:1233-1244` 切片前检查完整剩余请求的 KV 需求 · `:637-653` tile budget 首请求豁免 · `:1072` `add_one_req_ignore_eos` · `:841` `budget_state` · `:864` `_update_prefill_budget` |
| `managers/scheduler.py` | `:3428` 调用点 · `:3450` 身份比较判定是否被加入 |
| `mem_cache/radix_cache.py` | `:623-633` 加锁将节点从可驱逐集合转入受保护集合，影响准入余量 |

---
`schedule_policy.py:1208`，调用点 `scheduler.py:3428`。

**"被加入" = `self.can_run_list.append(req)` 被执行。**普通路径的两个直接 append 在
`:1371`（整条装得下）、`:1421`（切片）；`:1359` 的 dLLM 分支委托 `_add_dllm_req` 加入。

## 1. 九道闸（普通请求）

| # | 条件 | 不满足时 | 行 |
|---|---|---|---|
| 1 | 非 DSA-CP-in-seq-split 模式下的第二个请求 | `OTHER` | 1214 |
| 2 | `len(can_run_list) < prefill_max_requests` | `OTHER` | 1217 |
| 3 | `extend + max_new + page + mamba_gap < rem_total_tokens` | `NO_TOKEN` | 1243 |
| 4 | hybrid SWA 时 `swa_needed < rem_swa_tokens`（或触发 chunk cap 逃生） | `NO_TOKEN` | 1256 |
| 5 | 关 chunked prefill 时 `input_tokens < rem_input_tokens`（**本批第一个豁免**） | `OTHER` | 1270 |
| 6 | **加锁后** 条件 3、4 仍成立 | `NO_TOKEN` | 1282 |
| 7 | prefill delayer 放行 | `OTHER` | 1309 |
| 8 | AMD tile budget 未超（**本批第一个豁免**） | `OTHER` | 1355+ |
| 9 | 走切片路径时 `trunc_len > 0` 且满足 `truncation_align_size` | `OTHER` | 1390+ |

外加**隐含条件**（不在 `add_one_req` 里，但决定它能否被调用到）：LoRA 可调度（`scheduler.py:3381`）、
hicache 预取已完成（`:3400`）、`batch_is_full` 为 False 或抢占成功（`:3393`）、
在 `waiting_queue` 里排得够前（`:3316`）。

**闸 3 检查整个未缓存剩余输入及估计 decode 预留，不能用当前 chunk 大小替代。**
`cand_extend_input_len` 在切片前取完整剩余长度，`max_new` 是剩余生成上限经 `CLIP_MAX_NEW_TOKENS`
截断后的值；`rem_total_tokens` 还已扣掉 running 请求按 `new_token_ratio` 估计的未来 decode 占用。
这是容量预测，不是保证整个生成过程永不 retract。注意判失败用 `>=`，必须**严格小于**才过。

**开 chunked prefill 也不会跳过这道闸**：锁前 `:1243`、锁后 `:1282` 都检查原始 `total_tokens`，
到 `:1388` 才决定截断长度。切片控制本轮 forward 的工作量，并不普遍解除整请求 KV 准入导致的队首阻塞。

**`_lock_node` 那段为什么要重查闸 3、4**：`_lock_node` 对匹配到的 radix 节点 `inc_lock_ref`，
**防止分配新 KV 时触发 LRU 驱逐把自己刚匹配上的前缀驱逐掉**（自杀式驱逐）。
加锁会将原本可驱逐的前缀转为受保护状态，减少计入余量的 `evictable_size`，改变 `rem_total_tokens`——注释：
`self.rem_total_tokens may decrease after the lock acquisition`。

**闸 4（SWA）的两级判定**很精巧：

- `swa_needed >= rem_swa_tokens` 但 `_swa_req_never_fits()` 为 **False** → 只是**当前**挤，
  等 running 请求的窗口释放就能装下 → 返回 `NO_TOKEN` 让它等。硬塞进 decode headroom 会把
  SWA 可驱逐缓冲打穿，引发 "severe retraction/re-prefill storm"。
- `_swa_req_never_fits()` 为 **True**（按当前 chunk 配额算出的需求 `>= size_swa`，等池子排空也过不了闸）
  → 尝试 `_swa_chunk_cap()` 缩小 chunk；还要求已开启 chunk 且 cap 为正，否则仍返回 `NO_TOKEN`。
  这是特定 SWA 预算问题的逃生通道，不豁免前面的完整 KV 预算检查。

**闸 7（delayer）的位置是精心选的**：在所有 KV 闸之后（DP 各 rank 要如实上报自己能不能 prefill），
在 `init_load_back` 之前（决定要延迟了就不该开始搬 KV）。

## 2. 两处首请求豁免只针对计算量门槛

闸 5 和闸 8 都带 `len(self.can_run_list) != 0` 判断。注释写得很明白：

> if the can_run_list is empty, **always accept the first prefill request**

一个超长 prompt 如果每次都因"超出 `max_prefill_tokens`"被拒，会**永远卡在队首**。
所以本批第一个请求不因**这两道计算量门槛**被拒；完整 KV 预算、SWA、delayer 等其它检查仍然生效。
不能把注释里的 `always accept` 扩大成整个准入流程无条件接受。

## 3. 陷阱：返回值 ≠ 是否被加入

`add_one_req` 最后一行是 `return self.budget_state()`（`:1435`），它报的是
**"预算还够不够继续收下一个"**，不是"这个请求收没收"。

所以完全可能：**请求已经 append 进 `can_run_list`，但返回 `NO_TOKEN`**——因为正是这次
`_update_prefill_budget` 把预算扣到了 0。四种组合都真实存在：

| 是否 append | 返回值 | 含义 |
|---|---|---|
| ✅ | `CONTINUE` | 收了，还能继续收 |
| ✅ | `NO_TOKEN` | 收了，但 KV 预算被这次收人扣光了 |
| ✅ | `OTHER` | 收了，但 `max_prefill_tokens` / chunk 预算用完了 |
| ❌ | `NO_TOKEN` / `OTHER` | 没收，闸中途 return |

（不存在"没 append 却返回 `CONTINUE`"，因为提前 return 的分支只返回 `NO_TOKEN` / `OTHER`。）

调用方因此**只能用身份比较**（`scheduler.py:3450`）：

```python
added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
if not added:
    # 回收 init_next_round_input 里已分配的 mamba slot，否则泄漏
    ...mamba_allocator.free(req.mamba_pool_idx.unsqueeze(-1))
```

## 4. 通过之后的三条出路

| 出路 | 条件 | 关键差异 |
|---|---|---|
| **A: dLLM** | `dllm_config is not None` | `_add_dllm_req` |
| **B: 整条装下** | `chunk_tokens_limit is None or input_tokens <= chunk_tokens_limit` | `_update_prefill_budget` 传 `min(max_new_tokens, CLIP_MAX_NEW_TOKENS)` |
| **C: 切片** | 否则，且 `trunc_len > 0` | 传 **`max_new=0`**；设 `self.new_chunked_req = req` ★ |

出路 C 传 0 是因为 prompt 尚未喂完，本轮不会提交生成输出；**本次账本只扣当前 chunk，不扣该请求的 decode 预留**。
这不等于放弃首次准入的整请求容量检查：此前闸 3 已经用完整 `total_tokens` 检查过。
`new_chunked_req` 被调用方在 3479-3482 取走变成 `self.chunked_req`——**这就是 `chunked_req` 的诞生地**。

三条路都调 `_req_inc_lock_ref(req)`，把前缀节点的引用计数**持久化**（不同于 `_lock_node` 的临时锁）。

`_update_prefill_budget`（`:864`）**立即扣减**所有预算，所以**下一个请求看到的是扣减后的余额**，
而不是各自读取相同的初始余量：

```python
self.rem_total_token_offset += extend + max_new + page_overhead + mamba_gap    # 存量
self.cur_rem_token_offset   += extend +           page_overhead + mamba_gap    # 存量峰值
self.rem_input_tokens       -= extend                                          # 流量
if self.rem_chunk_tokens is not None:
    self.rem_chunk_tokens   -= extend                                          # 切片
```

## 5. `ignore_eos` 走另一套逻辑

条件是 `sampling_params.ignore_eos` **且** tree_cache 被禁用 → `add_one_req_ignore_eos`（`:1072`）。

这类请求**一定跑满 `max_new_tokens`**，乐观预留会失效。所以用**全局可行性检查**代替单请求预算：
构造 `req_states`（每个请求的 `(tokens_left, tokens_occupied)`，按 `tokens_left` 排序），
模拟"按结束先后顺序逐个释放"，检查任一时刻是否会耗尽 KV：

```python
for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
    bs = len(self.req_states) - i                      # 此刻还活着的请求数
    min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
    if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
        return AddReqResult.NO_TOKEN
    tokens_freed += tokens_occupied
```

`is_hybrid_swa` 时**跳过**这段模拟——SWA 的内存管理不同，这个机制会低估用量。

## 6. `new_token_ratio` 是个反馈环

用来估计 running 请求未来还会生成多少 token：发生 retraction 时**调高**预留（更保守），
稳定运行时 `decay_step()` **逐步变乐观**。

它避免了两个极端：始终按 `max_new_tokens` 全额预留（利用率太低），以及长期过度接纳（不停 retract）。

---

---

← [prefill 选批：`_get_new_batch_prefill_raw` 六步](06-prefill-admission.md)　|　[存量 vs 流量：两本 token 账](08-token-budget.md) →
