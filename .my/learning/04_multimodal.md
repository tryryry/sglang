# 阶段 4：多模态

## 任务 4.1：多模态请求的数据流
- [ ] 找到图片从请求到 pixel_values 的处理链路
- [ ] 读 `multimodal/processors/` 中某个 processor
- [ ] 理解 `grid_thw` 的含义（temporal, height, width 的 patch 网格）

**数据流笔记：**
```
图片 URL/base64
  → decode 成 PIL Image
  → image_processor (resize, normalize, split patches)
  → pixel_values tensor + grid_thw
  → vision encoder
  → image embeddings
  → 嵌入 text token 序列
  → LLM forward
```

---

## 任务 4.2：Vision Encoder 模型
- [ ] 读一个具体模型，如 `models/qwen2_vl.py`
- [ ] 找到 vision encoder 的调用位置
- [ ] 理解 `spatial_merge_size` / `merge_kernel_size` 的作用
- [ ] 理解 RoPE 3D（Qwen2.5-VL）vs RoPE 2D（Kimi-VL）

---

## 任务 4.3：Embedding 如何注入 LLM
- [ ] 找到 image embedding 如何替换 placeholder token
- [ ] 理解 multi-image 场景下多张图的 embedding 如何拼接
- [ ] 理解 `input_embeds` 是如何构造的

---

## 自检问题
1. `grid_thw = [1, 24, 24]` 表示什么？
2. 为什么需要 `embed_dim_reduction_factor`？
3. Vision encoder 的输出如何"插入"到文本序列中？
