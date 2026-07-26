# 实验 4:连续批处理与调度器观察(约 1.5 小时)

## 目标

用压测工具观察连续批处理(continuous batching)行为,理解调度器关键参数对吞吐与排队的影响。

## 步骤

### 1. 基准压测

启动服务器后,运行官方压测工具:

```bash
python -m sglang.bench_serving --backend sglang --port 30000 --num-prompts 200
```

记录关键指标:

- **Request throughput**(req/s)
- **Output token throughput**(tok/s)
- **Mean/median TTFT**(首 token 延迟)
- **Mean/median ITL**(token 间延迟)

### 2. 观察 prefill 与 decode 的动态混合

压测的同时观察服务器日志:

- `Prefill batch` 行:新请求进入,`#new-seq`、`#new-token`、`#cached-token`
- `Decode batch` 行:`#running-req`、`#token`、`token usage`、`gen throughput`

长短混合请求下,可以看到 prefill 和 decode 交替出现——这就是连续批处理:新请求随时插入,完成的请求随时退出。

### 3. 参数扫描

用本目录脚本对三个关键参数做对比实验(每组会重启服务器并压测一轮,耗时较长):

```bash
bash bench_params_sweep.sh Qwen/Qwen2.5-0.5B-Instruct 30000
```

对比的参数:

| 参数 | 作用 | 实验值 |
|---|---|---|
| `--max-running-requests` | 同时 decode 的最大请求数,太小会导致排队 | 32 / 256 |
| `--chunked-prefill-size` | 单次 prefill 的最大 token 数,控制 prefill 对 decode 的干扰 | 512 / 8192 |
| `--mem-fraction-static` | KV cache 池占显存比例,太小会导致频繁 retract | 0.5 / 0.85 |

### 4. 代码印证

阅读以下代码验证观察结果:

- `python/sglang/srt/managers/scheduler.py`:主事件循环(`event_loop_normal` / `event_loop_overlap`),看它如何每步组 batch、跑前向、处理结果;
- `python/sglang/srt/managers/schedule_policy.py`:从等待队列中选请求的策略(cache-aware 优先、FCFS 等)。

## 产出

- 不同参数组合下的 benchmark 对比报告(吞吐、TTFT、ITL 表格)。

## 思考题

1. `--max-running-requests` 调大一定会提升吞吐吗?什么情况下反而变差?
2. chunked prefill 解决了什么问题?如果没有它,一条超长 prompt 会对其它请求造成什么影响?
3. 日志中出现 "retract" 意味着什么?与 `--mem-fraction-static` 有什么关系?
