# 阶段 2：推理引擎核心流程

## 任务 2.1：请求入口
- [ ] 读 `srt/entrypoints/` 或 `srt/server.py`，理解 HTTP 接口
- [ ] 找到请求如何从 API 传递到 scheduler
- [ ] 记录关键函数调用链

**调用链笔记：**
```
请求 → ? → ? → ? → 返回
```

---

## 任务 2.2：调度器（Scheduler）
- [ ] 读 `srt/managers/scheduler.py`
- [ ] 理解 continuous batching：请求如何动态加入/移出
- [ ] 理解 prefill vs decode 阶段的区别

**关键概念：**
- Continuous Batching =
- Chunked Prefill =
- Waiting Queue vs Running Batch =

---

## 任务 2.3：模型执行
- [ ] 读 `model_runner.py`（或类似文件）
- [ ] 理解一次 forward 调用的输入/输出是什么
- [ ] 找到 TP 初始化的位置

---

## 任务 2.4：KV Cache 管理
- [ ] 读 `srt/mem_cache/` 相关文件
- [ ] 理解 RadixAttention 的核心思想（前缀树复用 KV）
- [ ] 理解 token 级别的内存分配

---

## 自检问题
1. 一个请求从到达到返回经过哪些组件？
2. Continuous batching 和传统 static batching 的区别？
3. RadixAttention 如何节省显存？
