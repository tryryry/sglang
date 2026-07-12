# 阶段 5：PR 审查实战

> 详细的 PR 笔记放在 [pr_review/](./pr_review/) 文件夹，每个 PR 一个文件。

## 任务 5.1：找到目标 PR
- [ ] 在 GitHub 上找到 liusiyu58 的 PR 列表
- [ ] 记录 PR 编号、标题、描述
- [ ] 本地 checkout PR 分支

**PR 列表：**
| PR # | 标题 | 状态 | 关注度 |
|------|------|------|--------|
| | | | |

**命令：**
```bash
# 方法1：GitHub CLI
gh pr list --author liusiyu58 --repo sgl-project/sglang

# 方法2：本地 git
git log --all --author="liusiyu58" --oneline

# 方法3：GitHub 网页搜索
# https://github.com/sgl-project/sglang/pulls?q=author:liusiyu58
```

---

## 任务 5.2：阅读 PR
- [ ] 读 PR 描述：解决什么问题、方案是什么
- [ ] 看改动文件列表，按依赖顺序排列阅读顺序
- [ ] 逐文件阅读 diff，记录疑问

**阅读笔记模板：**
```
PR #___: _______________

问题背景：

解决方案：

改动文件：
1. xxx.py → 做了什么
2. yyy.py → 做了什么

疑问：
- ❓ 
- ❓ 
```

---

## 任务 5.3：本地验证
- [ ] 切换到 PR 分支
- [ ] 跑相关单元测试
- [ ] 如果有性能改动，对比前后

```bash
git checkout pr-<NUMBER>
pytest test/srt/test_xxx.py -v
```

---

## 任务 5.4：总结与输出
- [ ] 用自己的话总结 PR 的核心改动
- [ ] 画改动前后的对比图
- [ ] 记录学到的设计模式或技巧

---

## 持续更新区

（每次看完一个 PR，在这里记录一行总结）

| 日期 | PR | 一句话总结 | 学到了什么 |
|------|-----|-----------|-----------|
| | | | |
