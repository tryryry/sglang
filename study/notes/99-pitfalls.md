# 最容易踩的坑

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`；下表列本次 chunked prefill 核查锚点，其它条目见各专题）

| 文件 | 关键位置 |
|---|---|
| `managers/schedule_policy.py` | `:1243/1282` 首次切片前检查完整 KV 预算 · `:637-653/1270-1278` 首请求只豁免计算量门槛 · `:1256-1268` SWA chunk cap 有条件启用 · `:1017-1020` hybrid SWA 续跑可停放 |
| `managers/scheduler.py` | `:3121-3132` 半截请求排除与条件 stash · `:3359-3361` 优先尝试续跑 · `:3437-3462` 非 CONTINUE 结束本轮扫描 |

---
| 坑 | 症状 | 根因 |
|---|---|---|
| `--dp N` 不开 dp-attention | 卡数比预期多一倍 | `dp_size` 是乘数还是除数，两种模式相反（[进程拓扑与请求分发](./02-process-topology.md)） |
| `batch_is_full` 漏复位 | 高并发下吞吐突然归零、有资源却不收新请求 | 跨轮持久的刹车没在批变小时清掉（[`get_next_batch_to_run` 主干](./04-scheduling-loop.md)） |
| merge 了 decode 的 `last_batch` | batch size 每步翻倍，KV 写冲突，一步就炸 | 少了 `is_extend()` 门卫（[四个正确性级别的机制](./05-batch-invariants.md)） |
| 半截 chunked 请求进了 decode | 状态错乱 | `finished()` 判断不了它，要靠黑名单（[四个正确性级别的机制](./05-batch-invariants.md)） |
| 不 filter 就 decode | 踩已释放的 KV slot | `finished` 的同一处已经 `release_kv_cache` 了（[四个正确性级别的机制](./05-batch-invariants.md)） |
| prefill-only 请求不清理 | `/v1/loads` 的 `num_running_reqs` 只增不减 | 常规收割点 `update_running_batch` 对它们永远不会被调用（[四个正确性级别的机制](./05-batch-invariants.md)） |
| 用返回值判断请求是否被加入 | mamba slot 泄漏 / 请求被漏掉 | `budget_state()` 报的是"还能不能继续"（[准入的核心：`add_one_req` 九道闸](./07-add-one-req.md)） |
| `total_tokens` 也减 `host_hit_length` | 载回那一刻 OOM | 搬运 ≠ 计算，存量维度一个格子都不能少（[存量 vs 流量：两本 token 账](./08-token-budget.md)） |
| 把 `input_tokens` 当成请求的固定属性 | 同一请求不同轮的准入结论对不上、复现不了 | 它随 cache 命中率逐轮变化，分块时还换函数算（[存量 vs 流量：两本 token 账](./08-token-budget.md) ⑩） |
| 显式传了 `--mem-fraction-static` 还指望调 chunk 能换 KV 池 | 调了没效果 | 推导分支外面包着 `if mem_fraction_static is None`，传了就断链（[KV 池与并发](./10-kv-pool-and-concurrency.md) §1） |
| 频繁 `Retract requests` | 吞吐抖动、重复 prefill | KV 池打满，看 `--mem-fraction-static` 和并发（[`get_next_batch_to_run` 主干](./04-scheduling-loop.md)） |
| 以为调小 `--chunked-prefill-size` 就能改善 decode ITL | 切了之后 ITL 没变好，吞吐还降了 | `chunked_req` 每轮无条件插队，prefill 恒胜；要让 decode 插进来必须另开 `--prefill-decode-interval` 或 `--enable-mixed-chunk`，两者默认都关（[Chunked prefill](./09-chunked-prefill.md) §1.1） |
| 以为开 chunk 或排本批第一就能绕过完整 KV 检查 | 长请求仍卡队首、后续短请求本轮不再尝试 | 首请求只豁免 input/tile 计算量门槛；首次切片仍先检查完整剩余输入加 decode 预留。SWA chunk cap 也只解决特定 SWA 需求 `>=` 整池容量的情形（[准入闸门](./07-add-one-req.md)、[Chunked prefill](./09-chunked-prefill.md)） |
| 把已有 chunk 的优先续跑写成无条件执行 | hybrid SWA 停放被误判成泄漏 | `_rem_tokens <= 0` 时可保留 `chunked_req` 而不加入本轮 batch；不变量是持续追踪状态且不提前进入 decode（[批次不变量](./05-batch-invariants.md)） |

---

---

← [Chunked prefill 的四条不变量](09-chunked-prefill.md)
