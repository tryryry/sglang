# 存量 vs 流量：两本 token 账

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/schedule_policy.py` | `:511` 构造 · `:671` `rem_total_tokens` (@property) · `:537-545` 本轮 input/chunk 配额 · `:841` `budget_state` · `:864` 扣预算 · `:1004-1053` chunk 续跑 · `:1243/1282` 切片前完整 KV 检查 |
| `managers/scheduler.py` | `:3005` stash 委托 cache · `:3131` 仅 stash 新增 KV · `:3340-3361` 每轮建账并先尝试续跑 chunk |
| `utils/common.py` | `:4468` `get_num_new_pages` 页数公式 |
| `mem_cache/multi_ended_allocator.py` | `:861` 页不够返回 None · `:865` `need_tokens` · `:2159` `available_size` 单位是 TOKEN |
| `mem_cache/radix_cache.py` | `:565-582` 半截尾页不进树，但纳入请求自己的 `prefix_indices` · `:623-633` 锁定前缀减少可驱逐量 |
| `mem_cache/chunk_cache.py` | `:89-94` 禁用 radix cache 时仍推进请求自己的 `prefix_indices` |

---

这是整个准入逻辑的地基，独立讲清楚。

## ① 一个 token 消耗两种物理资源

**资源 A：KV 格子。** 每个序列位置要存 K 和 V 两个向量：

```text
每 token 字节数 = 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

举例（28 层、4 个 KV head、head_dim=128、fp16）：`2×28×4×128×2 ≈ 56 KB/token`。
对普通 full attention，请求仍可能用到的历史 KV 会持续占用，是**存量**。
SWA 可以提前释放窗口外 KV；radix cache 也可能在请求结束后保留可复用前缀，不能一概认为结束即归还物理池。

**资源 B：forward 的宽度。** `input_ids` 长度为 N 的一次前向，materialize `[N, hidden]` 的
hidden states，跑 GEMM / attention / MLP。这些 activation 描述**本轮 forward 的工作集**，
buffer 可被后续步骤复用，不像完整历史 KV 那样随生成逐步累积——是**流量**。

## ② 画在时间轴上，差别一眼就出来

拿一个 prompt 800 token 全部新算、`max_new_tokens=128` 的普通 full-attention 请求来画。
下面是忽略页对齐、最后一个输出 token 的 KV 时序及共享缓存保留的概念图；**纵轴高度就是资源量**。

**流量**——每一步要推过网络多少 token；本轮工作量与后续各轮分别计量：

```text
 token
  800 ┤ ███
      │ ███
      │ ███
      │ ███    <- 柱高 = req.extend_range.length
      │ ███       input_tokens 是它页对齐后的记账值（≥ 柱高）
      │ ███
    1 ┤ ███   ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌
    0 └───────────────────────────────────────────> t
      prefill d1 d2 d3 d4 d5 d6 ...         d128
              └─────── 每根都只有 1 高 ────────┘
```

decode 那一串矮柱**不是** `add_one_req` 的 `input_tokens`——独立 decode 步不走这条 prefill 准入路径，
但仍有自己的内存检查。mixed chunk 模式还会在 `PrefillAdder` 构造时，先从本轮配额扣掉计划混入的 decode token。

**存量**——此刻请求需要持有多少 KV 格子。普通 full attention 会保留历史；图中结束后归零表示
请求不再持有这些状态，不等于 radix cache 一定同时释放全部物理页：

```text
 格子
  928 ┤                                      ╭────╮
      │                               ╭──────╯    │
      │                        ╭──────╯           │  <- 台阶峰值 = total_tokens
      │                 ╭──────╯                  │
      │          ╭──────╯                         │
  800 ┤   ╭──────╯                                │
    0 ┼───╯                                       ╰────> t
      prefill    d1     d2     d3     ...     d128  结束
          └─ 一次性开 800 格   └─ 每步 +1 格，只升不降
                                                  └─ 请求结束才全部归还
```

（峰值 928 = 800 + 128；闸门实际用的 `total_tokens` 再加一个 `page_size` 的余量。）

**同一件事写成数值**：

```text
时刻           prefill     d1     d2     d3    ...    d128    结束
──────────────────────────────────────────────────────────────────
流量 这步要算      800      1      1      1    ...       1       0
存量 此刻占着      800    801    802    803    ...     928       0
──────────────────────────────────────────────────────────────────
                    ▲                                   ▲
           input_tokens = 800                  total_tokens = 928
           （流量行的这一格）                   （存量行的峰值）
```

在这个简化的普通 full-attention 例子中：**流量逐步计量，历史 KV 沿生成累积。**

`max_new_tokens` 为什么只进 `total_tokens` 不进 `input_tokens`——现在一眼可见：那 128 个 token
在存量图里把台阶顶高了 128，在流量图里却是 **128 根各高 1 的独立柱子**，跟 prefill 那根柱子
根本不在同一个时刻。算力无法预支到这一步；显存却**必须现在就预留**，
否则 decode 到一半 KV 池空了就得 retract。

## ③ 核心：同一个 token，两个问题独立取值

对序列里任意位置 `i`，问两个**互相独立**的问题：

- **Q1**：它的 K/V 此刻在不在 device KV 池里？不在的话，谁为它开格子？
- **Q2**：这一步的 forward 要不要为它跑一遍 attention + MLP？

四种组合**全部真实存在**：

| 位置的来源 | Q1 要开格子？ | Q2 这次要算？ | 落在哪一项 |
|---|---|---|---|
| radix **device** 命中的前缀 | ❌ 已经占着（共享别人的） | ❌ | 被 `prefix_indices` 扣掉，两边都不进 |
| HiCache **host** 命中的前缀 | ✅ 要新开（搬回来） | ❌ 搬运不过 attention | 只进 `total`，被 `input` 减掉 |
| 真正没命中的新 token | ✅ | ✅ | 两边都进 |
| **未来** decode 的 token | ✅ 要预留 | ❌ 现在它还不存在 | `max_new`，只进 `total` |

**这张表就是全部答案。**两个数是同一张表两列打勾项的求和：

```text
total_tokens  = 第二列的和 = cand_extend_input_len + max_new + page_size
                              └─ 含 host 命中那段 ─┘
input_tokens  = 第三列的和 = ceil_page(cand_extend_input_len − host_hit_length)
```

两个数衡量不同成本，不能把数值大小直接理解成资源的包含关系。第 2 行和第 4 行都是"只占不算"，但原因不同：
**host 命中是"已存在但不需重算"，`max_new` 是"还不存在所以无从算起"。**

## ④ 四个变量

其实是**四个**量，不是两个：

| 变量 | 行 | 公式 | 维度 | 用在哪 |
|---|---|---|---|---|
| `cand_extend_input_len` | 1230 | `len(full_untruncated_fill_ids) - len(prefix_indices)` | 原料 | 派生下面三个 |
| `total_tokens` | 1233 | `cand + max_new + page_size + mamba_gap` | **存量** | 锁前、锁后的 KV 总预算检查 |
| `real_input_tokens` | 1239 | `ceil_page(cand - host_hit_length)` | **流量** | SWA 需求计算、锁前 input 配额检查 |
| `input_tokens` | 1332 | `ceil_page(len(full_untruncated_fill_ids) - len(prefix_indices))`，锁内重算 | **流量** | 锁内 input 配额检查、tile budget、切片判定、扣预算 |

**页对齐方式也不同**：`total_tokens` 直接**加一个 `page_size`**，`input_tokens` 是 `ceil_page()`
**向上取整**到记账粒度；实际 forward 长度仍取 `extend_range.length`，不保证等于这个对齐值。
前者需要单独解释。

### 那个 `+ page_size` 是什么

**分配器实际扣的不是 `extend_len`**——KV 按页发，不按 token 发：

```python
# utils/common.py:4487  get_num_new_pages
num_new_pages = ceil(seq_lens / page_size) - ceil(prefix_lens / page_size)
# multi_ended_allocator.py:865
need_tokens   = num_new_pages * page_size        # ← 真正从池子里扣掉的
```

尾页浪费无法避免：radix tree 里能共享的只有**完整页**，请求末尾那个填不满的页专属它自己。
`cache_unfinished_req` 的注释直说了：

> for `page_size > 1`, the partial part is added to `req.prefix_indices`,
> but **that part of kv indices is not added to the tree**

**最坏浪费恰好是 `page_size − 1`。** 设 `prefix_len = aP + r`（`P` = page_size，`e` = extend_len）：

| 前缀对齐 | 新开页数 | 说明 |
|---|---|---|
| `r = 0` | `ceil(e/P)` | 从新页开头写，尾页浪费 `ceil_page(e) − e` |
| `r > 0` | `ceil((r+e)/P) − 1` | 先把尾页剩的 `P−r` 个空位**白捡**，再开新页 |

两种情况都 `≤ ceil_page(e) ≤ e + P − 1`。扫遍所有对齐位置验算（`P = 64`）：

```text
extend=  1   最坏实际占用=  64   超出 e = 63
extend= 63   最坏实际占用=  64   超出 e =  1
extend= 64   最坏实际占用=  64   超出 e =  0
extend= 65   最坏实际占用= 128   超出 e = 63
extend=100   最坏实际占用= 128   超出 e = 28
extend=300   最坏实际占用= 320   超出 e = 20
```

**加一整个 `page_size` 正好盖住，还富余 1 个格子。**

**为什么"每请求一页"能覆盖整个生命周期**（而不是"每次 extend 留一页"）——请求最终 `seq_len = p+e+m`：

```text
一辈子新开的页 = ceil((p+e+m)/P) − ceil(p/P) ≤ ceil_page(e+m) ≤ (e+m) + P − 1
而 total_tokens =                                                (e+m) + P
```

一个 `+P` 盖住 prefill 加全部 decode，decode 阶段不必再单独留。这跟 `alloc_decode` 的行为对得上：
它只在 `seq_lens % page_size == 1` 时开新页（`common.py:4484`），每 `P` 步才开一次。

**为什么不精确算**——`len(prefix_indices) % P` 是已知的，精确算得出来。但 `max_new_tokens` 本身
就是上界猜测（真实生成长度未知），为一个零头做精确化没有意义；固定 `+P` 是 O(1) 的安全上界。
代码自己的 TODO 承认了这点：`may be too conservative`。

扣预算时其实**双重保守**：`_update_prefill_budget` 先把 extend 做 `ceil_paged_tokens()` 向上取整，
**然后又加一个 `page_overhead = page_size`**（`:877-884`）——ceil 已经盖住 extend 的零头，`+P` 再兜一层。

**少留会怎样**——`alloc_extend` 在页不够时直接 `return None`（`:861`），而这时 batch 已经组好、
请求已经被准入放行：**准入说行、分配说不行**。这道余量就是防这个的保险。

（`page_size = 1` 时退化成 `+1`，几乎无成本。这个预留只在 paged 布局下才有分量，
如 MLA / FlashInfer 的 `page_size = 64`。）

## ⑤ 结构证据：代码里两者的类型就不一样

这是最硬的一条。

```python
@property
def rem_total_tokens(self):                                  # :671
    available_and_evictable = (
        self.token_to_kv_pool_allocator.available_size()     # ← 读分配器的真实状态
        + self.tree_cache.evictable_size()                   # ← 读 cache 的真实状态
    )
    return available_and_evictable - self.rem_total_token_offset
```

```python
self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens    # :537
#                       └─ 构造时传入的 self.max_prefill_tokens，一个配置常量
```

- **`rem_total_tokens` 是 `@property`，每次访问都去测量物理现实**——是**读数**。
- **`rem_input_tokens` 是普通实例变量，从配置常量 `max_prefill_tokens`（默认 16384）初始化再一路减**
  ——是**配额**。

这就是为什么 `_lock_node` 之后必须重查 KV 总预算和 SWA 预算：加锁将前缀从可驱逐集合转为受保护集合，
**可供准入使用的容量读数会变**。input 配额本身不会被锁修改，但代码仍在 load-back 后用重算的
`input_tokens` 再检查一次该配额（`:1336-1343`）。

## ⑥ 两个返回值的语义由此而来

`budget_state()`（`:841`）把这层区别直接写成了两个结果码：

```python
no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
if no_token:
    return AddReqResult.NO_TOKEN          # 物理资源见底
if self.rem_input_tokens <= 0:
    return AddReqResult.OTHER             # 本轮配额用完，资源还在
```

- `NO_TOKEN` = **某项容量预算不够**，可能是可用 KV、SWA、Mamba slot 或未来占用预留不足，
  不代表物理显存已经全部用完；请求结束、前缀状态或估计预留改变后可能恢复。
- `OTHER` = 本轮配额或其它准入限制要求停止；不保证下一轮立即可收，
  也不保证能通过那一轮的容量和 delayer 检查。

**跨轮继承的是资源状态，不是同一个 adder**：每次选批重建 `PrefillAdder`，input/chunk 配额重置，
总量 offset 也根据当时 running 请求重算；allocator/cache 里已经占用的 KV 状态则会跨轮保留。

## ⑦ 为什么不能合成一本账

因为同一个 token 在两个维度的成本**比例不固定，且随缓存状态实时变化**
（下表 prompt 均为 1000 token，`page_size=64`）：

| 场景 | `total_tokens` | `input_tokens` | 比例 |
|---|---|---|---|
| 全没命中，`max_new=128` | 1192 | 1024 | ≈1.2 |
| radix 命中 900，`max_new=128` | 292 | 128 | ≈2.3 |
| host 命中 700，`max_new=128` | 1192 | 320 | ≈3.7 |
| 全没命中，`max_new=4096` | 5160 | 1024 | ≈5.0 |

比例从 1.2 摆到 5.0，**任何单一标量都无法同时约束这两件事**。两种失败模式也不同：

| 算错哪本 | 失败模式 |
|---|---|
| `input_tokens` 放太多 | 这次 forward 太宽 → 延迟尖刺、decode ITL 被打爆、activation buffer 溢出 |
| `total_tokens` 放太多 | KV 池被打爆 → retract → 重复 prefill → 吞吐雪崩 |

## ⑧ `input_tokens` 为什么要算两次

锁前（1239）是**预估**——"假设载回会成功，我要算这么多"；
锁内（1332）是**实测**——`init_load_back` 真的把 host KV 搬回并并入 `prefix_indices` 了，
那段从"要算的"变成"已在 prefix 里的"。**锁内重算不是冗余，是用事实替换预估。**

## ⑨ 一个数字例子

prompt 1000，radix device 命中 200，HiCache host 命中 300，`max_new_tokens=128`，`page_size=64`：

```text
cand_extend_input_len = 1000 - 200 = 800       # device 上没有的部分
host_hit_length       = 300                     # 其中 300 在 host，可搬不用算

total_tokens      = 800 + 128 + 64 = 992                        # 存量：800 格都要占（含要搬的 300）
real_input_tokens = ceil_page(800 - 300) = ceil_page(500) = 512 # 流量：只算 500，对齐到 512

load_back 后 prefix_indices 从 200 涨到 500：
input_tokens = ceil_page(1000 - 500) = 512                      # 与 real_input_tokens 一致

扣预算：rem_total_token_offset += 512 + 128 + 64 = 704          （比闸门用的 992 小）
        rem_input_tokens      -= 512
```

同一个请求，存量维度记 992（闸门）/ 704（扣减），流量维度记 512。**三个数都对，因为问的是不同的问题。**
闸门用的数偏大（含 host_hit、未页对齐），宁可拒绝也不冒险放行——正是 `_update_prefill_budget`
那句 TODO 的自我评价 `may be too conservative`。

## ⑩ `input_tokens` 不是一个静态的数

它不是请求的属性，是 **(请求 × 这一刻的 cache 状态 × 这一次准入调用)** 的函数。三个层次都会让它变。

**层次一：同一次 `add_one_req` 调用内就算两次**——见 ⑧，锁前预估 / 锁内实测。

**层次二：同一个请求、不同调度轮次，值不同。** 这是最本质的一层：

```python
cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
```

分子固定，但 `prefix_indices` 每轮由 `init_next_round_input(self.tree_cache)` **重新做一次前缀匹配**得出。
别的请求刚把这段前缀写进 radix tree → 这轮命中率高 → `input_tokens` 骤降；LRU 把它驱逐了 → 又涨回来。
**同一个请求在队列里等了 3 轮，这 3 轮的 `input_tokens` 可以完全不同**——这正是 [prefill 选批：`_get_new_batch_prefill_raw` 六步](./06-prefill-admission.md) ④ 那句
"同一个请求在不同时刻能否加入的结论可能完全不同"的出处。

**层次三：首次 chunk 准入后，后续续跑走 `add_chunked_req`。**成功推进时剩余长度减小；
hybrid SWA 停放的轮次没有进展。

```text
轮 1     add_one_req      → input_tokens > chunk_limit → 出路 C 切片 → 成为 chunked_req
轮 2..N  add_chunked_req  ← 这个函数里根本没有 input_tokens 这个变量
```

首次 `add_one_req` 在决定切片前仍检查完整剩余 `total_tokens`（`:1243/1282`）；
所以 chunk 不会普遍解除整请求 KV 容量门槛。

后续 `add_chunked_req`（`:1004`）不用 `add_one_req` 那套新请求闸门，但有自己的预算逻辑：
普通路径先取 `_rem_tokens = min(rem_chunk_tokens, int(rem_total_tokens))`；hybrid SWA 再限制到
`int(rem_swa_tokens) - page_size`，若结果非正，直接返回原请求、本轮不加入 batch。
非 hybrid-SWA 的普通路径才在结果非正时回退到 `rem_chunk_tokens`。
然后用剩余长度和 `_rem_tokens` 的较小值设置 `extend_range`；这里没有复用首次切片的
`truncation_align_size` 检查，不能把两条路径的截断细节混为一谈。

`prefix_indices` 在后续调度步处理前一块时推进：`stash_chunked_request` →
`maybe_cache_unfinished_req` → 所用 cache 的 `cache_unfinished_req`。
仅有新增 KV（`extend_range.end > len(prefix_indices)`）才 stash，停放轮不重复 stash。
普通 radix 路径**重写 `req.prefix_indices`** 覆盖已 prefill 的部分：

```python
# radix_cache.py:574 注释；完整页分支赋值在 :582
# `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
req.prefix_indices = new_indices
```

禁用 radix cache 时，`ChunkCache.cache_unfinished_req` 也会从请求的 KV 索引表复制已处理部分
到 `prefix_indices`（`chunk_cache.py:89-94`）；**续跑所需的请求内前缀复用不依赖跨请求 radix 缓存**。
成功推进后，下一轮的 `cand_extend_input_len` 自然变短。

`rem_chunk_tokens` 是整个 batch 共用的预算：构造时 mixed decode 先扣一份（`:544-545`），
每个加入的请求再扣页对齐后的 extend（`:874/903`），不是给每条请求单独分配固定大小的 chunk。

**另外：`input_tokens` 也不等于实际喂进 forward 的长度。**那是 `req.extend_range` 的长度：

| 出路 | `set_extend_range` | 实际长度 |
|---|---|---|
| B 整条装下（:1368） | `(len(prefix_indices), len(full_untruncated_fill_ids))` | `cand_extend_input_len`，**未页对齐** |
| C 切片（:1417） | `(len(prefix_indices), len(prefix_indices) + trunc_len)` | `trunc_len` |

整条路径中 `input_tokens = ceil_page(cand_extend_input_len) ≥ 实际长度`；
原始长度已页对齐时即可相等，不要求 `page_size=1`。
出路 C 更直接——扣预算传的是 `trunc_len`，`input_tokens` 在那条路上只用来判"装不装得下"。

---

← [准入的核心：`add_one_req` 九道闸](07-add-one-req.md)　|　[Chunked prefill 的四条不变量](09-chunked-prefill.md) →
