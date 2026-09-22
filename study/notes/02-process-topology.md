# 进程拓扑与请求分发

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `entrypoints/engine.py` | `:818` `_launch_scheduler_processes` 分叉点 · `:1805` `_calculate_rank_ranges` 多机切分 |
| `managers/data_parallel_controller.py` | `:88` LoadBalanceMethod · `:372` launch_dp_schedulers · `:554` dp-attn 分支 · `:747` 外部路由口子 |
| `managers/scheduler_components/request_receiver.py` | `:104` `_pull_raw_reqs` · `:153` `_broadcast_reqs_across_ranks` |
| `server_args.py` | `:6987` tp_size 即 world size · `:6992` dp-attn 约束 · `:4238` load-balance auto 解析 |

---
## 1. 一个 Scheduler 进程 = 一张 GPU

Scheduler 不是"一个"，是一组。每张参与推理的 GPU 对应一个独立进程，各自持有自己的
`waiting_queue` / `running_batch` / `TpModelWorker` / `ModelRunner`。

**模式 A：经典 DP（不开 `--enable-dp-attention`）**

```text
Scheduler 进程总数 = dp_size × pp_size × tp_size
```

每个 DP replica 是**一份完整的模型副本**，各自独占 `tp_size × pp_size` 张卡，互不通信。
`launch_dp_schedulers`（`data_parallel_controller.py:372`）循环 `dp_size` 次，每次把起始卡号往后推。

**模式 B：DP attention（`--enable-dp-attention`）**

```text
Scheduler 进程总数 = pp_size × tp_size        ← dp_size 不再是乘数
约束：tp_size % (dp_size × attn_cp_size) == 0        (server_args.py:6992)
派生：attn_tp_size = tp_size // dp_size // attn_cp_size
```

这里 `dp_size` 是**切分** `tp_size` 而不是复制它。attention 按 DP 切（每组独立处理自己那批请求），
MoE/MLP 仍按全局 TP/EP 切。`server_args.py:6987` 的注释点破关键：

> The tp_size is the world size, not the real tensor parallel size

**以 `--tp 8 --dp 2 --pp 1` 为例：**

| 配置 | Scheduler 进程数 | GPU 数 | 拓扑 |
|---|---|---|---|
| 不开 dp attention | 2 × 1 × 8 = **16** | 16 | 两份完整模型副本，各占 8 卡 |
| 开 dp attention | 1 × 8 = **8** | 8 | 一份模型，attention 切 2 个 DP group，每组 attn_tp=4 |

**差出一倍的卡，这是最容易踩的坑。**

进程由谁创建（`entrypoints/engine.py:818` 是分叉点）：

```text
dp_size == 1:
    TokenizerManager ──ZMQ──→ Scheduler(tp0..tp3)     直接 fork tp × pp 个

dp_size > 1:
    TokenizerManager ──ZMQ──→ DataParallelController ──ZMQ PUSH──→ 各 DP 组
                                   （多一层进程，由它再 fork）
```

多机切分见 `_calculate_rank_ranges`（`engine.py:1805`）：`--tp 16 --nnodes 2` → node0 拿 `tp_rank` 0-7、
node1 拿 8-15；`--tp 8 --pp 2 --nnodes 2` → 每机一个 PP stage，两机 `tp_rank` **都是 0-7**
（`tp_rank` 是组内编号，全局位置由 `(pp_rank, tp_rank)` 确定）。

## 2. 请求分发是两级的

```text
TokenizerManager
      │  ZMQ PUSH (TokenizedGenerateReqInput)
      ↓
┌──────────────────────────────────────────┐
│ 第一级：DataParallelController           │  仅 dp_size > 1 时存在
│   四种 LoadBalanceMethod 选一个 DP rank  │  进程间，ZMQ PUSH
└──────────────────────────────────────────┘
      │  送到该 DP replica 的 tp_rank 0
      ↓
┌──────────────────────────────────────────┐
│ 第二级：TP rank 0 收，广播给同组其它 rank│  组内，gloo broadcast_pyobj
└──────────────────────────────────────────┘
      ↓
  该组每个 Scheduler 拿到一模一样的请求列表
```

**第一级的四种策略**（`data_parallel_controller.py:88`）：

| 策略 | 做法 | 默认用在 |
|---|---|---|
| `round_robin` | 在**活跃** worker 里轮询，跳过不可用的 | 非 PD、PD decode |
| `follow_bootstrap_room` | `bootstrap_room % len(workers)` | PD prefill |
| `total_requests` | 选 `running + waiting` 最少的 | — |
| `total_tokens` | 选 token 最少的，请求数做 tie-break | — |

`follow_bootstrap_room` 用取模而不是负载，是因为 PD 分离下**同一请求的 P 侧和 D 侧必须落在配对的 rank 上**
——KV 要从那个 prefill 节点传到那个 decode 节点。**确定性哈希是硬要求，不是优化。**

后两种从**共享内存快照**读负载（不是实时 RPC），并带本地启发式：派完一个就先把该 rank 负载 +1
（`:136`），否则同批请求会因读到同一份快照而全打到同一个 rank。

另有绕过所有策略的口子 `maybe_external_dp_rank_routing`（`:747`）：`req.routed_dp_rank` 非空则直送。

**第二级为什么必须广播**：同一 TP 组内的 Scheduler 必须做出**完全相同**的调度决策，否则各 rank
的 batch 形状不一致，forward 里的 all-reduce 直接对不上。做法不是指望各自收到一致，而是
**只让一个 rank 收然后广播**（`request_receiver.py:104` / `:153`）。

work req 和 control req **分开走**是这里的关键设计：

- **work req**（推理请求）：DP attention 下只需在 `attn_tp_group` 内广播。
- **control req**（profile、权重更新、block/unblock）：所有 rank 都得收到，必须走全 `tp_group`。

除非开 `--enable-dp-attention-local-control-broadcast`——那时 DP controller 给**每个** group leader
都发一份（`control_message_step = 1`），控制消息也能只在本组广播，省掉一次昂贵的全局 gloo 同步。

## 3. 四种通信手段别混

| 通道 | 手段 | 传什么 |
|---|---|---|
| Tokenizer → Scheduler | ZMQ (ipc://) | `TokenizedGenerateReqInput` 等 |
| Scheduler rank0 → rank i | gloo `broadcast_pyobj` | 收到的请求列表（Python 对象），控制面 |
| Scheduler ↔ Scheduler | NCCL | forward **内部**的 all-reduce/all-gather，**不是** Scheduler 层发起的 |
| Scheduler → TpModelWorker | 普通函数调用 | `ForwardBatch` |
| Scheduler rank0 → Detokenizer | ZMQ | 输出 token |

## 4. 三个 "rank" 别混

| 名字 | 含义 | 谁在用 |
|---|---|---|
| `tp_rank` | TP 组内编号 0 ~ `tp_size-1` | 决定 gpu_id |
| `attn_tp_rank` | DP attention 下 attention TP 组内编号 | **决定是否负责收 work req** |
| `dp_rank` | 第几个 DP replica / attention DP group | DP controller 路由目标 |
| `pp_rank` | 流水线第几段 | 只有 `pp_rank == 0` 从 ZMQ 收，其余从上一段点对点收 |

收请求的判据是 `attn_tp_rank == 0 and attn_cp_rank == 0`，**不是** `tp_rank == 0`。
经典模式下两者等价；DP attention 下每个 DP group 都有自己的 `attn_tp_rank == 0`，各收各的。

---

---

← [请求的一生：从 HTTP 到 Req](01-request-lifecycle.md)　|　[调度器的五个状态变量](03-scheduler-state.md) →
