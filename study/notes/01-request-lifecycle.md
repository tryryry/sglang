# 请求的一生：从 HTTP 到 Req

> SGLang 学习笔记 · [返回总索引](../README.md)

**对应源码**（根目录 `python/sglang/srt/`）

| 文件 | 关键位置 |
|---|---|
| `entrypoints/http_server.py` | HTTP 入口，`ORJSONRequest` |
| `entrypoints/openai/serving_chat.py` | `:1100` 原始 messages · `:1386` 渲染 chat template · `:1394` prompt_ids · `:966` 构造内部请求 |
| `entrypoints/openai/protocol.py` | `:823` `ChatCompletionRequest` 的**完整字段定义**——想知道某个参数叫什么、默认值是多少，看这里而不是抓包 |
| `managers/tokenizer_manager.py` | `:768` 请求标准化 · `:959` input_ids 直通 · `:1334` 生成 TokenizedGenerateReqInput |

---
```text
HTTP JSON
  ↓ 解析、schema 校验、补默认值、嵌套对象转换
ChatCompletionRequest
  ↓ chat template 渲染 + tokenize
TokenizedGenerateReqInput
  ↓ ZMQ / IPC
Scheduler.handle_generate_request()
  ↓
Req
  ↓
waiting_queue
  ↓ 排序（policy）+ 准入（PrefillAdder）
ScheduleBatch (EXTEND)
  ↓
TpModelWorker → ModelRunner → GPU forward
  ↓ ZMQ
DetokenizerManager → HTTP response
```

## `raw_request` vs `ChatCompletionRequest`

两者不重复，回答的是不同问题：

| | 回答什么 | 内容 |
|---|---|---|
| `raw_request` | 这次 HTTP 连接是谁、从哪来、是否仍连着 | headers、URL、method、client、连接状态、原始 body |
| `ChatCompletionRequest` | 模型应该怎样生成 | 校验+补默认值+类型转换后的生成参数 |

```text
raw_request.body() + schema 校验 + 默认值 + 类型转换 = ChatCompletionRequest
```

**Scheduler 不直接处理原始 HTTP JSON**。它收到的是 frontend 已标准化并 tokenize 过的内部请求，
再构造统一的 `Req` 进入调度系统。

关键落点（Python frontend 路径）：

| 位置 | 做什么 |
|---|---|
| `serving_chat.py:1100` | 拿到原始 messages |
| `serving_chat.py:1386` | 渲染 chat template |
| `serving_chat.py:1394` | 得到 `prompt_ids` |
| `serving_chat.py:966` | 构造内部请求 |
| `tokenizer_manager.py:768` | 请求标准化 |
| `tokenizer_manager.py:1334` | 生成最终 Tokenized 请求 |

---

---

← [Debug 速查：启动、请求、断点](00-debug-playbook.md)　|　[进程拓扑与请求分发](02-process-topology.md) →
