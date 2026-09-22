# Chunked prefill：为什么切、代价、四条不变量

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/schedule_policy.py` | `:1233-1243` 闸 2 用**整条**未缓存长度 · `:1388` 才算 `trunc_len` · `:1004` `add_chunked_req`（`:1017` SWA 停放 / 非 SWA 兜底）· `:1417` 切片出路设 `new_chunked_req` · `:1270` 闸 5 只在关掉 chunk 时生效 |
| `managers/scheduler.py` | `:3359` `chunked_req` 无条件插队 · `:1216` `_should_defer_prefill`（**无 chunk 豁免**）· `:3462` 非 CONTINUE 直接 break · `:3005` `stash_chunked_request` · `:3121-3132` 加黑名单 · `:3008` pending abort |
| `mem_cache/radix_cache.py` | `:516` `cache_unfinished_req` 重写 `prefix_indices` |
| `server_args.py` | `:832` `--chunked-prefill-size` · **`:5425-5432` 决定 `reserved_mem`（在 KV 池之前扣）** · `:5160-5240` 按显存分档的默认值 · `:7288` `max_prefill_buffer_tokens` · `:841` `prefill_decode_interval` 默认 0 · `:1026` `enable_mixed_chunk` 默认 False · `:4034/4104/4174/5123` 四处强制关闭 |
| `model_executor/runner/eager_runner.py` | `:117` buffer registry 按 `max_prefill_buffer_tokens()` 建 |

---

## 0. 前提：为什么能切

**causal attention 下，前缀的 KV 算完就固定不变**——后面的 token 看得见前面的，前面的看不见后面的。

所以把 prompt 切成几段顺序算，和一次算完，结果**逐位相同**。这是 chunked prefill 唯一的硬前提，
也是它成立的全部理由。前提不成立的场景（双向注意力、encoder-only）会被强制关掉，见 §4。

```text
chunk 1 → chunk 2 → ... → final chunk → decode
```

---

## 1. 切能带来什么

先把话说清楚：**只有 §1.2 是 chunk 无条件带来的；§1.1 需要额外开关配合；§1.3 是准入语义的变化。**

### 1.1 decode ITL：chunk 是**必要不充分**条件

一次 forward 只能是一种 `ForwardMode`。一个 100k token 的 prefill 跑多久，所有正在 decode
的请求就一个 token 都出不来。正常 decode step 约 10–30 ms，一次超大 prefill 可能几百 ms 到数秒。

**但"切了就能让 decode 插进来"是错的。**默认配置下切完照样不让：

```python
# _get_new_batch_prefill_raw 开头，scheduler.py:3359
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)   # 无条件插队
```

半截请求每轮抢先进 `can_run_list` → `new_batch` 恒非空 → 主干第 ④ 步 **prefill 恒胜**
（[主干](./04-scheduling-loop.md)）→ decode 分支根本到不了。**chunk 背靠背跑，
总 stall 与不切基本相同，还多了重读 KV 的开销。**

让路的判断在更上游，且**没有 `chunked_req` 豁免**：

```python
# scheduler.py:3184-3190
elif self._should_defer_prefill():      # :1216，短路在 get_new_batch_prefill 之前
    new_batch = None                    # → 走 decode
```

```python
def _should_defer_prefill(self) -> bool:            # :1216
    if self._prefill_decode_interval_remaining == 0:
        return False                                # ← 默认恒走这里
    self._prefill_decode_interval_remaining -= 1
    return True
```

而 `_arm_prefill_decode_interval`（`:1224`）在**任何 extend 批**跑完后重置计数器，chunk 也算。
所以真正让 decode 插进来的是这两个开关，**默认都是关的**：

| 开关 | 默认 | 机制 |
|---|---|---|
| `--prefill-decode-interval N` | **0（关）** | 每跑完一个 chunk，强制 N 轮 decode 才允许下一个 chunk |
| `--enable-mixed-chunk` | **False（关）** | decode 请求并入每个 chunk 的 extend 批，同一次 forward 出 token |

**chunk 的贡献是创造了"可以让路的时刻"**：不切的话整个 prefill 是一次已发射的 kernel，
调度器物理上无法中途介入；切成 N 段才有 N 个调度点。但**要不要让路由上面两个开关决定**。

> 想靠 chunk 改善 ITL，必须同时开 `--prefill-decode-interval` 或 `--enable-mixed-chunk`。
> 只调小 `--chunked-prefill-size` 而两个开关都不开，ITL 不会变好。

（`prefill_delayer` 帮不上忙：`add_chunked_req` 的注释明写
"A mid-chunk rank prefills this pass **regardless of the delayer verdict**"。）

### 1.2 静态 activation 预留变小 → KV 池变大 → 并发变高

**这是 chunked prefill 唯一无条件成立、且可量化的好处。**

`chunked_prefill_size` 直接决定 activation 预留，而这块是**在 KV 池之前扣掉**的：

```python
# server_args.py:5425-5432（另一处等价逻辑在 :5318-5330）
elif self.chunked_prefill_size > 0:
    activation_tokens = max(self.chunked_prefill_size, 2048)
else:
    activation_tokens = max(self.max_prefill_tokens, 2048)      # 关掉 chunk 时退回这里
reserved_mem = 512 + activation_tokens * 1.5 + self.tp_size * self.pp_size / 8 * 1024
```

代入（`tp=8, pp=1`，单位 MB）：

| 配置 | `activation_tokens` | `reserved_mem` |
|---|---|---|
| `--chunked-prefill-size 2048` | 2048 | ≈ **4.5 GB** |
| `8192`（H100 默认） | 8192 | ≈ **13.5 GB** |
| `-1` 关闭 → 退回 `max_prefill_tokens`（默认 16384） | 16384 | ≈ **25.5 GB** |

**在 80 GB H100 上，开 chunk(8192) 比关掉多出约 12 GB 给 KV 池**——按 56 KB/token 估算
约 21 万个 KV 格子。

> 关键认识：chunk 省的**不是"运行时的峰值"，而是"静态预留"**。
> 它让你敢把这块预留调小，省下的显存全部流向 KV 池。

`reserved_mem` 通过 `mem_fraction_static` 一路影响到 `max_total_num_tokens` 和并发，
完整链条、**两个会让这条链失效的前提**（显式传 `--mem-fraction-static`、post-capture sizing）
以及"并发为什么值钱"，见 [KV 池与并发](./10-kv-pool-and-concurrency.md)。

这也解释了为什么默认值**按显存档位定、不按算力**（`server_args.py:5160-5240`）：

| GPU 显存 | 典型卡 | 默认 `chunked_prefill_size` |
|---|---|---|
| < 20 GB | T4、4080 | 2048 |
| < 35 GB | A10、4090、5090 | 2048 |
| < 60 GB | A100-40G、L40 | 4096 |
| < 90 GB | H100、A100-80G | 8192 |
| < 160 GB | H20、H200 | 8192 |
| ≥ 160 GB | B200、MI300 | 16384 |

**它本来就是个显存参数，ITL 是顺带的（而且还要另开开关，见 §1.1）。**

### 1.2.1 推论：长 prompt 才可服务

单次 forward 的 activation 不再随 prompt 长度增长，prompt 长度的上限从
"activation 显存"解耦出来，只剩 KV 池容量一个约束（也就是 §2 的闸 2）。

eager runner 的 buffer registry 也按这个数建：

```python
prefill_ceiling = max(mr.max_total_num_tokens, max_prefill_buffer_tokens())   # eager_runner.py:117
```

而 `max_prefill_buffer_tokens()`（`server_args.py:7288`）的定义就是
"Prefill-buffer ceiling: `chunked_prefill_size`"。

### 1.3 算力闸从"逐请求拦"变成"固定宽度切片"

关掉 chunk 时，超长 prompt 要过闸 5（`max_prefill_tokens`）：

```python
if (self.rem_chunk_tokens is None          # ← 只有关掉 chunked prefill 才成立
    and len(self.can_run_list) != 0
    and real_input_tokens >= self.rem_input_tokens):
    # if the can_run_list is empty, always accept the first prefill request
    return AddReqResult.OTHER              # schedule_policy.py:1270
```

注意第一个条件 `rem_chunk_tokens is None`——**开着 chunked prefill 时这道闸整个被跳过**，
因为切片本身已经把单次规模限住了。

关掉之后，超长 prompt 只能靠 `len(can_run_list) != 0` 这个"本批第一个无条件接受"的豁免破例进来。
代价是：这一批**就它一个**，而且这一次 forward 就是个超大 batch——§1.1 和 §1.2 全部触发。
**豁免不是解法，是没有 chunk 时的补救。**

一个容易过度解读的点：闸 5 被跳过 ≠ `max_prefill_tokens` 失效。`budget_state()`（`:851`）里的

```python
if self.rem_input_tokens <= 0:
    return AddReqResult.OTHER
```

还在，所以它仍是**批级**上限，只是不再逐请求拦截。

---

## 2. ⚠️ chunk 解决不了什么：KV 总量闸照旧拦整条

**这是最容易搞错的一点，两本预算必须拆开看。**

闸 2 检查的是全条未缓存 prompt，跟切不切无关：

```python
cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
total_tokens = cand_extend_input_len + max_new + self.page_size      # :1233
...
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN                                     # :1243（锁内 :1282 再查一次）
```

`cand_extend_input_len` 是**整条**未缓存长度；`chunk_tokens_limit` 要到 `:1361` 才用于分支、`:1388` 才算出 `trunc_len`。
同一结论在 [九道闸](./07-add-one-req.md) 的闸 3 处也有说明。
也就是说：

> **请求必须先证明"整条的 KV 装得下"，才有资格被切。**
> KV 放不下的请求，开不开 chunked prefill 都一样返回 `NO_TOKEN`。

而且闸 2 **没有"本批第一个"豁免**（闸 5、闸 8 才有）。配合主循环对非 `CONTINUE` 的处理：

```python
if res != AddReqResult.CONTINUE:
    if res == AddReqResult.NO_TOKEN:
        running_batch.batch_is_full = True
    ...
    break                      # scheduler.py:3462
```

是 `break` 不是 `continue`——**队首一个 KV 装不下的巨型请求，会把后面所有短请求这一轮一起堵死，
而 chunked prefill 对此无能为力。**

这类队首阻塞的解药是另外三样，都不在 chunk 这条线上：

| 手段 | 位置 |
|---|---|
| tree cache 驱逐 | 已计入 `rem_total_tokens = available + **evictable** − offset`（`:671`） |
| `retract_decode` 抢占回退 | `scheduler.py:3635`，把 running 请求踢回队列腾 KV |
| 优先级抢占 | `priority_scheduling_preemption_threshold` → `adder.preempt_list` |

### 2.1 chunk 对 KV 的间接好处（确实有，但是另一回事）

**① 扣款不对称：闸门要全条，实际只扣切片。**

| 路径 | 闸 2 要求 | `_update_prefill_budget` 实扣 |
|---|---|---|
| 整条（出路 B，`:1375`） | `cand + max_new + page` | `ceil_page(cand) + max_new + page` |
| 切片（出路 C，`:1425`） | `cand + max_new + page`（**一样**） | `trunc_len + **0** + page` ← `max_new` 传 0 |

切片请求这一轮不生成 token，不必为它预留 decode 空间。所以它进来之后**留给后面请求的余额多得多**
——受益的是同批的其它请求，不是它自己。

**② 轮 2..N 不再走闸 2。** 后续 chunk 走 `add_chunked_req`（`:1004`），它不调用 `add_one_req`，
而是自己算 `_rem_tokens = min(rem_chunk_tokens, rem_total_tokens)` 后直接截断：

```python
if _rem_tokens <= 0:
    if self.is_hybrid_swa:
        return req                          # ← 停放：保留 chunked_req，本轮不加入 batch
    _rem_tokens = self.rem_chunk_tokens      # ← 非 SWA：兜底给一份，避免半截卡死泄漏
```

注意**不是无条件硬塞**：hybrid SWA 下预算见底会 `return req` 把请求**停放**——
调用方 `self.chunked_req = adder.add_chunked_req(...)` 收下这个返回值，状态继续被追踪，只是本轮不跑。
非 SWA 路径才兜底给一份 `rem_chunk_tokens`，注释说明理由是半截请求卡住会造成内存泄漏。

**闸 2 只在第一次准入时拦一道**，但这不等于后续 chunk 完全无约束（见 §4 不变量 2）。

**③ 唯一的 chunk-based 预算逃生通道在 SWA，不在 full KV 池。** 闸 3 里：

```python
if swa_needed >= self.rem_swa_tokens:
    if not self._swa_req_never_fits(...):
        return AddReqResult.NO_TOKEN          # 只是暂时挤，等
    swa_cap = self._swa_chunk_cap(...)        # 永远装不下 → 缩小 chunk 逃生
    chunk_tokens_limit = min(self.rem_chunk_tokens, swa_cap)
```

SWA 池遇到"怎么等都装不下"会**缩小 chunk 放行**，full KV 池**没有对应物**。
这个不对称本身就说明：用切片绕开 KV 闸是可以做的，但目前只在 SWA 那条路上做了。

---

## 3. 切的代价：为什么不无脑切小

| 代价 | 说明 |
|---|---|
| HBM 带宽 | 每个 chunk 的 attention 都要重读前面**全部** KV，chunk 越多重读次数越多 |
| GPU 利用率 | 大 GEMM 拆成小 GEMM，kernel launch 变多，chunk 太小时算不满 |
| 总 FLOPs | **不降反略升**——切开后 attention 总量仍是 O(n²)，还多了调度和读写开销 |
| 状态复杂度 | 引入 `chunked_req`，带来下面 §4 的一整套不变量 |

所以 `--chunked-prefill-size` 是纯粹的 tradeoff 旋钮：**小 → ITL 好、吞吐差；大 → 吞吐好、ITL 差。**

---

## 4. 四条不变量

1. 未完成的 `chunked_req` **不能提前进入 `running_batch`**，因为它还没完成 prefill
   （[四个正确性级别的机制](./05-batch-invariants.md)）。
2. 它必须在下一轮**优先续跑**——`add_chunked_req` 绕开普通门禁，但**不是无条件执行**：
   非 SWA 路径在 `_rem_tokens <= 0` 时兜底给一份 `rem_chunk_tokens`（注释理由是半截卡住会内存泄漏），
   hybrid SWA 则 `return req` 把它**停放**到下一轮。不变量是"持续追踪状态、不提前进 decode"，
   不是"每轮必须跑"。
3. 前一 chunk 完成后要 stash KV 进 radix cache（`cache_unfinished_req` 会**重写 `req.prefix_indices`**
   覆盖已 prefill 完的那段），但**只有 final chunk 完成后请求才能进入稳定 decode**。
4. overlap 下用 inflight 计数追踪尚未处理完成的中间 chunk；abort 也必须延迟到调度步开头的安全点
   （`process_pending_chunked_abort`），不能就地拆。

**代价的本质**：把"要么全收要么不收"变成"收多少算多少"，换来一个必须在后续轮次优先续上的 `chunked_req`。

---

## 5. 什么时候会被强制关掉

代码里四处 `chunked_prefill_size=-1`，**全是"切开算结果就错了"或"根本不走 paged KV"**，不是性能选择：

| 位置 | 场景 | 原因 |
|---|---|---|
| `server_args.py:4034` | HRM text 模型且 `prefix_lm` | 递归 forward + **双向注意力**，切开语义就变了 |
| `:4104` | EmbeddingGemma（encoder-only） | 单次 prefill 走 raw-K/V 快路径，**不读写 paged KV cache** |
| `:4174` | `is_multimodal_chunked_prefill_supported` 为 False 的多模态模型 | 视觉 token 不能从中间切断 |
| `:5123` | `--enable-mis`（multi-item scoring） | 与该特性不兼容 |

第一条最能说明问题：**双向注意力下前缀的表示会被后文改写，§0 那个前提不成立，chunk 就不再是等价变换。**

---

## 一句话总结

> Chunked prefill 做的只有一件事：**给单次 forward 的宽度加上限**。由此：
>
> | | |
> |---|---|
> | **无条件得到** | **静态 activation 预留变小 → KV 池变大 → 并发变高**（§1.2）。chunk=8192 与关闭相比在 H100 上约差 12 GB；省的是**静态预留**，不是运行时峰值 |
> | **需要额外开关** | decode ITL——chunk 只创造了让路的时刻，让不让路看 `--prefill-decode-interval` / `--enable-mixed-chunk`，两者默认都关（§1.1） |
> | **完全不解决** | "整条请求 KV 放不下"——闸 2 拦的是全条未缓存 prompt，切片要过了这道闸才发生（§2） |
>
> 前提是 causal attention 让分段计算等价于一次计算（§0）。

---

← [存量 vs 流量：两本 token 账](08-token-budget.md)　|　[KV 池与并发](10-kv-pool-and-concurrency.md) →
