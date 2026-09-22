# 调度器的五个状态变量

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `managers/schedule_batch.py` | `:3464` `NextBatchPlan` · `:3322` `merge_batch` |
| `managers/scheduler.py` | `:1159` running_batch 初值 · `:1163` last_batch · `:3538` mixed chunk 清空 · `:4611` `collect_inflight_reqs`（PP 下用 `running_mbs` 而非 `running_batch`） |

---
| 状态 | 含义 | 生命周期 | 能否为 None |
|---|---|---|---|
| `waiting_queue` | 已进 scheduler、尚未获准 prefill | 可跨很多轮 | — (list) |
| `running_batch` | 已完成 prefill、正在 decode 的集合 | **跨轮持久** | 不能（空时是 `reqs=[]` 的空 batch） |
| `last_batch` | 上一轮**实际执行**的 batch | 只保留一轮 | 能 |
| `batch_to_run` | 本轮立即送进模型的 batch | **一步，用完即弃** | 能（空转时） |
| `chunked_req` | prompt 未喂完、下轮必须续跑的长请求 | 跨多个 prefill 轮 | 能 |

一句话区分：

```text
batch_to_run  = 这一轮做什么          （瞬时的动作）
running_batch = 下一轮仍需保留哪些 decode 请求  （持久的状态）
last_batch    = 上一轮做了什么        （一步的记忆，用于毕业判断）

传送带：batch_to_run ──(本轮末尾)──→ last_batch ──(下轮 merge)──→ running_batch
```

`running_batch` 空闲时是空 batch 而不是 `None`——这样 `batch_is_full`、`is_empty()` 不用到处判空。

## `batch_to_run` 与 `running_batch` 的四种关系

| 场景 | `batch_to_run` | `running_batch` | 关系 |
|---|---|---|---|
| **Prefill 轮** | `{A,B}` 新 extend 批 | `{C,D}` 老 decode 稳态 | **两个不同对象** |
| **Decode 轮** | `{C,D}` | `{C,D}` | **同一个对象**（`update_running_batch` 原地改并返回自己） |
| **空转** | `None` | `{}` 空批 | 一个 None 一个空批 |
| **Mixed chunk** | `{A,B,C,D}` extend+decode 混在一个 forward | `{}` 被清空 | decode 请求"借"给了 `batch_to_run` |

Mixed chunk 那种最反直觉：C、D 这一步的 decode 在 `batch_to_run` 里完成，`running_batch` 若还留着它们，
下一轮会**重复调度**。注意清空用的是**重新构造一个 `ScheduleBatch` 只继承 `batch_is_full`**，
而不是原地 `reqs.clear()`——符合 `.claude/rules/schedule-batch-out-of-place-mutation.md`。

## 为什么 merge 要推迟一轮

`NextBatchPlan` 是**纯函数式**返回（`schedule_batch.py:3464`）：

```python
class NextBatchPlan(msgspec.Struct):
    batch_to_run: Optional[ScheduleBatch]
    running_batch: ScheduleBatch
```

调用方**必须两个都接住**，少接一个就丢状态（mixed chunk 的清空、prefill 的 merge 全丢）。
历史上是直接改 `self.running_batch`，改成显式传入传出有两个原因：

1. **PP 多 microbatch**：PP 开启时用的是 `self.running_mbs`，每个 microbatch 各持一份 decode 状态。
2. **可测试 / 无副作用**：输入输出都在签名里。

而 merge 本身推迟到下一轮开头，是因为 **overlap 模式下 prefill 的结果要到下一轮才落地**，
必须等一轮才能判断哪些 req 真的完成了（finished / 还在 chunk 中）。

---

---

← [进程拓扑与请求分发](02-process-topology.md)　|　[`get_next_batch_to_run` 主干](04-scheduling-loop.md) →
