# SGLang 学习笔记

对 [sglang](https://github.com/sgl-project/sglang) 源码的阅读笔记。按主题拆成独立小文件，
**每篇开头都有源码映射表**，可以对着代码看。

- 行号对应 `study/2026-08-26` 分支当时的代码，读的时候以函数名为准、行号为辅。
- 内容源自 2026-09 的一轮源码精读（原始流水笔记已并入本目录，见文末「出处」）。
- 读主干时，先把 `if self.enable_xxx`（hisparse / dllm / lora / hicache / spec / dp-attn / ngram）
  **全部当作 False 跳过**，主干清楚了再逐个打开。

---
我在 study/ 下维护 SGLang 的源码学习笔记（README.md 总索引 + notes/ 专题）。

回答我关于这个仓库的问题时，默认把结论写回笔记，不要只在对话里说，也不要问我要不要写：

1. 判断结论属于 notes/ 的哪一篇。已有对应主题就补充或修订那篇；确实是新主题就新建一篇，
   按现有编号规则命名，并在 README 的目录表和代码地图里各加一行。
2. 如果这次讨论证伪了笔记里的某个说法、或发现原来的措辞不准确（哪怕只是用词松），一并改掉，
   不要留着不一致。
3. 每篇笔记开头必须有「对应源码」表：路径相对 python/sglang/srt/，带行号和一句话说明。
   跨篇引用用相对链接，不要用章节号。
4. 结论要有代码依据。引用行号前先 grep 确认存在，端点和 CLI 参数先验证再写进去，
   不确定的标注出来，不要凭印象写。
5. 回答我时，先讲清楚结论和推导，最后简短说明改了哪几个文件的哪几处。

## 目录

| # | 笔记 | 一句话 | 主要源码 |
|---|---|---|---|
| 00 | [Debug 速查](notes/00-debug-playbook.md) | 启动命令、curl、断点清单、怎么故意触发某条路径 | `entrypoints/http_server.py`、`server_args.py` |
| 01 | [请求的一生](notes/01-request-lifecycle.md) | HTTP JSON 怎么变成 scheduler 里的 `Req` | `entrypoints/`、`managers/tokenizer_manager.py` |
| 02 | [进程拓扑与请求分发](notes/02-process-topology.md) | 起几个 Scheduler、请求怎么分到它们手上 | `entrypoints/engine.py`、`managers/data_parallel_controller.py` |
| 03 | [调度器的五个状态变量](notes/03-scheduler-state.md) | `waiting_queue` / `running_batch` / `last_batch` / `batch_to_run` / `chunked_req` | `managers/schedule_batch.py` |
| 04 | [`get_next_batch_to_run` 主干](notes/04-scheduling-loop.md) | 一轮调度的四段：清理 → 毕业 → 选 prefill → 退化 decode | `managers/scheduler.py:3095` |
| 05 | [四个正确性级别的机制](notes/05-batch-invariants.md) | 门卫、黑名单、收割器、prefill-only——少一个就出错 | `managers/scheduler.py`、`schedule_batch.py` |
| 06 | [prefill 选批六步](notes/06-prefill-admission.md) | `_get_new_batch_prefill_raw` 怎么组出一个 extend batch | `managers/scheduler.py:3267` |
| 07 | [`add_one_req` 九道闸](notes/07-add-one-req.md) | 一个请求要满足什么条件才能被收进 prefill 批 | `managers/schedule_policy.py:1208` |
| 08 | [存量 vs 流量：两本 token 账](notes/08-token-budget.md) | `total_tokens` 管 KV 驻留、`input_tokens` 管前向宽度 | `managers/schedule_policy.py`、`mem_cache/` |
| 09 | [Chunked prefill](notes/09-chunked-prefill.md) | 为什么要切、**它解决不了什么**（KV 闸照旧拦整条）、代价、四条不变量 | `managers/schedule_policy.py:1004`、`server_args.py` |
| 10 | [KV 池与并发](notes/10-kv-pool-and-concurrency.md) | 显存预留怎么一路变成并发上限；并发为什么值钱、代价是什么 | `mem_cache/kv_cache_configurator.py`、`server_args.py` |
| 99 | [最容易踩的坑](notes/99-pitfalls.md) | 症状 → 根因 → 对应章节 | — |

**手上有环境、想边跑边看**，先开 [00](notes/00-debug-playbook.md)。

**通读顺序**：01 → 02 建立全局观 → 03 → 04 看主循环 → 05 理解不变量 → 06 → 07 → 08 深入准入 → 09 → 10 回到容量。
只想查某个具体问题，直接看 [99](notes/99-pitfalls.md)。

---

## 速查：十条最重要的结论

| # | 结论 | 展开 |
|---|---|---|
| 1 | 一个 Scheduler 进程 = 一张 GPU。同样写 `--dp 2`，开不开 `--enable-dp-attention` 差一倍的卡 | [02](notes/02-process-topology.md) |
| 2 | 同 TP 组内各 Scheduler 的 `waiting_queue` / `running_batch` 内容**完全相同**——不是巧合，是 rank 0 收了再广播 | [02](notes/02-process-topology.md) |
| 3 | **prefill 优先是硬编码的**。所有延迟控制都靠在组批前提前 `return None`，而不是改优先级 | [04](notes/04-scheduling-loop.md) |
| 4 | 新 prefill batch 本轮执行，**下一轮**才通过 `last_batch` 毕业进 `running_batch`；merge 必须推迟一轮 | [03](notes/03-scheduler-state.md) |
| 5 | decode 轮里 `batch_to_run` 与 `running_batch` 是**同一个对象**，再 merge 一次就是灾难 | [05](notes/05-batch-invariants.md) |
| 6 | `batch_is_full` 是跨轮持久的刹车，凡"批变小"处都必须复位，否则高并发下永久不再接新请求 | [04](notes/04-scheduling-loop.md) |
| 7 | Policy 只**排序**，`PrefillAdder` 才**准入**；`PrefillAdder` 只**预测**，`prepare_for_extend()` 才**分配** | [06](notes/06-prefill-admission.md) |
| 8 | `total_tokens` 是**存量**（占用曲线峰值）、`input_tokens` 是**流量**（这一次 forward 的宽度），两者没有包含关系 | [08](notes/08-token-budget.md) |
| 9 | `add_one_req` 的**返回值不代表请求是否被加入**，调用方只能用身份比较 | [07](notes/07-add-one-req.md) |
| 10 | KV 不足的兜底是 retract 不是 OOM；日志里 `Retract requests` = 容量告警 | [04](notes/04-scheduling-loop.md) |

---

## 代码地图：从源码反查笔记

根目录 `python/sglang/srt/`。

| 源码 | 关键位置 | 讲它的笔记 |
|---|---|---|
| `entrypoints/http_server.py` | HTTP 入口 | [01](notes/01-request-lifecycle.md) |
| `entrypoints/openai/serving_chat.py` | `:966` `:1100` `:1386` `:1394` | [01](notes/01-request-lifecycle.md) |
| `managers/tokenizer_manager.py` | `:768` `:1334` | [01](notes/01-request-lifecycle.md) |
| `entrypoints/engine.py` | `:818` 起进程 · `:1805` 多机切分 | [02](notes/02-process-topology.md) |
| `managers/data_parallel_controller.py` | `:88` `:372` `:554` `:747` | [02](notes/02-process-topology.md) |
| `managers/scheduler_components/request_receiver.py` | `:104` 收 · `:153` 广播 | [02](notes/02-process-topology.md) |
| `server_args.py` | `:847` `max_prefill_tokens` · `:6987` `:6992` dp-attn | [02](notes/02-process-topology.md) · [08](notes/08-token-budget.md) |
| | `:832` `chunked_prefill_size` · `:5160-5240` 按显存分档 · `:4034/4104/4174/5123` 强制关闭 | [09](notes/09-chunked-prefill.md) |
| `managers/scheduler.py` | `:3095` `get_next_batch_to_run` | [04](notes/04-scheduling-loop.md) |
| | `:3267` `_get_new_batch_prefill_raw` | [06](notes/06-prefill-admission.md) |
| | `:3581` `update_running_batch` · `:1216` defer | [04](notes/04-scheduling-loop.md) |
| | `:3005` `stash_chunked_request` · `:3008` pending abort | [09](notes/09-chunked-prefill.md) |
| `managers/schedule_batch.py` | `:3182` `filter_batch` · `:1225` `is_prefill_only` | [05](notes/05-batch-invariants.md) |
| | `:3464` `NextBatchPlan` · `:3322` `merge_batch` | [03](notes/03-scheduler-state.md) |
| | `:1307` `init_next_round_input` · `:1426` 匹配上限 | [08](notes/08-token-budget.md) |
| `managers/schedule_policy.py` | `:511` `PrefillAdder` · `:671` `:841` `:864` 预算 | [08](notes/08-token-budget.md) |
| | `:1208` `add_one_req` · `:1072` `ignore_eos` | [07](notes/07-add-one-req.md) |
| | `:1004` `add_chunked_req` | [09](notes/09-chunked-prefill.md) |
| `model_executor/forward_batch_info.py` | `:104` `ForwardMode` | [05](notes/05-batch-invariants.md) |
| `managers/tp_worker.py` | `:658` prefill-only 不采样 | [05](notes/05-batch-invariants.md) |
| `managers/scheduler_components/batch_result_processor.py` | `:327` 释放 KV · `:410` 假 token | [05](notes/05-batch-invariants.md) |
| `mem_cache/radix_cache.py` | `:516` `cache_unfinished_req` · `:576` 半截尾页不进树 | [08](notes/08-token-budget.md) · [09](notes/09-chunked-prefill.md) |
| `mem_cache/multi_ended_allocator.py` | `:861` 页不够返回 None · `:2159` 单位是 TOKEN | [08](notes/08-token-budget.md) |
| `utils/common.py` | `:4468` `get_num_new_pages` | [08](notes/08-token-budget.md) |
| `mem_cache/kv_cache_configurator.py` | `:1827` `_profile_available_bytes` · `:1936` `resolve_max_num_reqs` · `:1999-2002` 字节→token→并发 | [10](notes/10-kv-pool-and-concurrency.md) |
| `server_args.py` | `:5309` 仅在未显式指定时推导 · `:5336-5343` `mem_fraction_static` · `:5425-5432` activation 预留 | [10](notes/10-kv-pool-and-concurrency.md) |
| `disaggregation/prefill.py`、`disaggregation/decode.py` | `:546` / `:2555` PD 两侧入口 | [04](notes/04-scheduling-loop.md) |

---

## 读码最短路径

```text
event_loop_normal / event_loop_overlap        scheduler.py:1778 / 1822
  ↓
get_next_batch_to_run                         scheduler.py:3095      → 笔记 04
  ↓
get_new_batch_prefill
  ↓
_get_new_batch_prefill_raw                    scheduler.py:3267      → 笔记 06
  ↓
PrefillAdder.add_one_req                      schedule_policy.py:1208 → 笔记 07 / 08
  ↓
ScheduleBatch.prepare_for_extend              schedule_batch.py:2408
  ↓
run_batch
  ↓
process_batch_result
```

---

## 还没覆盖的模块

这份笔记目前只走完了**调度器**这条线。以下是 SGLang 里同等重要、但还没动的部分：

| 模块 | 路径 | 关注点 |
|---|---|---|
| 注意力后端 | `layers/attention/` | FlashInfer / Triton / FA3 的 metadata 构造与 CUDA graph 捕获 |
| KV cache 与前缀复用 | `mem_cache/` | radix tree、分页分配器、HiCache 三级（device/host/storage） |
| 模型执行 | `model_executor/` | `ModelRunner` 的 forward 组织、CUDA graph、piecewise capture |
| 投机解码 | `speculative/` | EAGLE / draft-extend / target-verify 与调度器的交互 |
| MoE 与专家并行 | `layers/moe/` | dispatch/combine、EP 与 DP attention 的配合 |
| PD 分离 | `disaggregation/` | bootstrap room、KV 跨节点传输 |
| 权重加载与量化 | `model_loader/`、`layers/quantization/` | FP8 / MXFP8、权重更新 |
| 采样与约束解码 | `sampling/`、`constrained/` | grammar 后端、logit processor |
| LoRA | `lora/` | adapter 调度与 batch 内多 adapter |
| 多模态 | `multimodal/` | 图像 token 化与 `image_tokens` 计费 |
| 模型定义 | `models/` | 各模型的 forward 实现与权重映射 |
| 分布式原语 | `distributed/` | 通信组、custom all-reduce |
| 图编译 | `compilation/` | piecewise / torch.compile 路径 |
| 专家负载均衡 | `eplb/` | EP rebalance |
| 路由层 | 仓库根 `rust/`、`sgl-model-gateway/` | 多实例前置路由（独立于 srt） |

---

## 目录结构

```text
study/
├── README.md        ← 本文件，总索引
├── notes/           按主题拆分的笔记，每篇带源码映射表
└── demo/            IPC 演示脚本：simple_ipc_demo.py、scheduler_ipc_demo.py
```

## 出处

这些笔记提炼自 2026-09-10 ~ 09-20 的原始流水记录（`study/LOG`、`LOG_9_1`、`LOG_9_10`、
`LOG_9_16`、`LOG_9_17`、`LOG_9_20`，共约 3600 行）。结论、命令和源码锚点都已并入 `notes/`，
原始文件于 2026-09-22 删除。需要回查时：

```bash
git show 07eab796ac:study/LOG_9_16   # 调度主干 + 附录 A1-A8 + 进程拓扑 B1-B6
git show 07eab796ac:study/LOG_9_20   # add_one_req 九道闸
git show f5748fb3a9:study/LOG_9_17   # _get_new_batch_prefill_raw 逐行走读
git show f5748fb3a9:study/LOG_9_1    # HTTP 入口原始记录
git show f5748fb3a9:study/LOG        # 上一版整理稿
git show d788f5762e:study/LOG_9_10   # 进程拓扑原始图
```
