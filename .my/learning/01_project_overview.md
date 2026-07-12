# 阶段 1：项目全貌

## 任务 1.1：理解项目定位
- [ ] 读 README.md，理解 SGLang 是什么、解决什么问题
- [ ] 对比 vLLM / TGI，理解差异点（RadixAttention、前端语言）
- [ ] 浏览 `docs/` 目录，了解文档结构

**笔记区：**
```
（写你的理解）
```

---

## 任务 1.2：目录结构
- [ ] 浏览顶层目录，理解各文件夹职责
- [ ] 重点看 `python/sglang/srt/` 下的子目录
- [ ] 画出自己的目录结构脑图（可以很粗略）

**关键目录：**
```
python/sglang/srt/
├── distributed/        → 通信原语
├── multimodal/         → 多模态处理
├── models/             → 模型实现
├── layers/             → 自定义层
├── managers/           → 调度器
├── mem_cache/          → KV Cache
└── server.py           → 服务入口
```

---

## 任务 1.3：环境搭建与运行
- [ ] 创建虚拟环境，安装依赖
- [ ] 尝试启动一个小模型（如果有 GPU）
- [ ] 或者：只读代码，跳过实际运行

**命令备忘：**
```bash
pip install -e ".[all]"
python -m sglang.launch_server --model-path <model> --tp 1
```

---

## 自检问题
1. SGLang 的核心优势是什么？
2. `srt` 目录下最重要的 5 个子目录是？
3. 启动服务的入口文件是哪个？
