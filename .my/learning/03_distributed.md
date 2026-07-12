# 阶段 3：分布式并行

## 任务 3.1：通信基础
- [ ] 读 `srt/distributed/__init__.py` 和 `parallel_state.py`
- [ ] 理解 TP group 如何创建
- [ ] 理解 `get_tensor_model_parallel_rank()` / `world_size()` 的含义

**笔记：**
```
TP group =
rank =
world_size =
```

---

## 任务 3.2：通信原语
- [ ] 读 `srt/distributed/communication_op.py`
- [ ] 理解 `tensor_model_parallel_all_gather` 的实现
- [ ] 理解 `tensor_model_parallel_all_reduce` 的实现
- [ ] 画图：2 GPU 上 all-gather 前后数据变化

**对比：**
| | All-Gather | All-Reduce |
|---|---|---|
| 输入 | 每卡一份不同数据 | 每卡一份不同数据 |
| 输出 | 拼接所有数据 | 所有数据的 SUM |
| 输出大小 | N 倍 | 不变 |
| TP 中用于 | ? | ? |

---

## 任务 3.3：TP 在线性层中的应用
- [ ] 读 `srt/layers/linear.py`（或 `vocab_parallel_embedding.py`）
- [ ] 理解 ColumnParallelLinear：权重按列切分
- [ ] 理解 RowParallelLinear：权重按行切分
- [ ] 理解为什么 Column → Row 组合后需要 all-reduce

**示意图：**
```
ColumnParallel:  W 按列切 → 各卡算部分输出 → 拼接(all-gather)
RowParallel:     W 按行切 → 各卡算部分和 → 求和(all-reduce)
```

---

## 任务 3.4：Encoder DP（数据并行）
- [ ] 读 `mm_utils.py` 的 `get_dp_encoder_lb_assignment`
- [ ] 理解贪心负载均衡算法
- [ ] 读 `run_dp_sharded_mrope_vision_model` 完整流程
- [ ] 理解"借 TP 的壳做 DP"的设计

**关键变量追踪表：**
| 变量 | 含义 | 示例值 |
|------|------|--------|
| `patches_per_image` | | |
| `image_to_tp_rank` | | |
| `cum_patches_per_image` | | |
| `embed_dim_reduction_factor` | | |
| `grouped_pixel_values_len` | | |

---

## 自检问题
1. TP 和 DP 分别切什么？
2. All-gather 和 all-reduce 分别在什么场景使用？
3. 为什么 vision encoder 用 DP 而不用 TP？
4. 负载均衡分配中为什么大图优先？
