"""
SGLang Scheduler IPC 通信最小复刻 demo.

单文件复现 sglang 三个进程之间的 ZMQ 通信拓扑:

    主进程 (HTTP server + TokenizerManager)
      | PUSH bind  scheduler_input_ipc_name      TokenizedGenerateReqInput 等请求
      v
    Scheduler 子进程 (每个 TP rank 一个)
      | PUSH connect  detokenizer_ipc_name       BatchTokenIDOutput (token id)
      v
    DetokenizerManager 子进程
      | PUSH connect  tokenizer_ipc_name         BatchStrOutput (文本)
      v
    主进程 TokenizerManager (PULL bind tokenizer_ipc_name, 按rid匹配, 唤醒 Future)

对照真实源码:
  PortArgs.init_new               python/sglang/srt/server_args.py:10657
  TokenizerManager 的 socket      python/sglang/srt/managers/tokenizer_manager.py:549
  Scheduler 的 socket             python/sglang/srt/managers/scheduler_components/ipc_channels.py:24
  DetokenizerManager 的 socket    python/sglang/srt/managers/detokenizer_manager.py:113
  sock_send / sock_recv (msgpack) python/sglang/srt/managers/io_struct.py:2433
  NOBLOCK 排空式收请求            python/sglang/srt/managers/scheduler_components/request_receiver.py:118
  event_loop_normal               python/sglang/srt/managers/scheduler.py:1765
  stream_output                   python/sglang/srt/managers/scheduler_components/output_streamer.py:114
  DetokenizerManager.event_loop   python/sglang/srt/managers/detokenizer_manager.py:169
  TokenizerManager.handle_loop    python/sglang/srt/managers/tokenizer_manager.py:2154

运行 (仓库根目录, pyzmq/msgspec 是 sglang 自带依赖):
  ./.venv/bin/python study/scheduler_ipc_demo.py
"""

import asyncio
import dataclasses
import multiprocessing as mp
import os
import tempfile
import time
import uuid
from typing import Optional, Union

import msgspec
import zmq
import zmq.asyncio

# ============================================================
# 1. 消息定义 —— 对应 io_struct.py:
#    BaseReq = msgspec.Struct, tag=True, kw_only=True (io_struct.py:82)
#    tag=True 让每条消息自带类型标签, 接收端用 Decoder(Union[...]) 一步还原。
# ============================================================


class BaseReq(msgspec.Struct, tag=True, kw_only=True):
    rid: str


class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True):
    pass


class TokenizedGenerateReqInput(BaseReq):
    """TokenizerManager -> Scheduler: tokenized prompt"""

    input_ids: list[int]


class BatchTokenIDOutput(BaseBatchReq):
    """Scheduler -> DetokenizerManager: 每 step 新产出的 token id (增量)"""

    rids: list[str]
    new_token_ids: list[list[int]]
    finished: list[bool]


class BatchStrOutput(BaseBatchReq):
    """DetokenizerManager -> TokenizerManager: 解码出的增量文本"""

    rids: list[str]
    deltas: list[str]
    finished: list[bool]


class FlushCacheReqInput(BaseReq):
    """TokenizerManager -> Scheduler: 控制消息与生成请求走同一条通道"""

    flush_cache_id: int


class FlushCacheReqOutput(BaseReq):
    """Scheduler -> TokenizerManager: 控制回包, 直接走 send_to_tokenizer, 不经过 detokenizer"""

    flush_cache_id: int


class ShutdownReq(BaseReq):
    pass


_ALL_TYPES = Union[
    TokenizedGenerateReqInput,
    BatchTokenIDOutput,
    BatchStrOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    ShutdownReq,
]

# 对应 io_struct.py:2389-2392
_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_decoder = msgspec.msgpack.Decoder(_ALL_TYPES)


def sock_send(sock: zmq.Socket, obj: object) -> None:
    """对应 io_struct.py:2433 —— 默认 msgpack; sglang 里还有 SGLANG_USE_PICKLE_IPC 的 pickle 分支"""
    sock.send(_msgpack_encoder.encode(obj))


def sock_recv(sock: zmq.Socket, flags: int = 0) -> object:
    """对应 io_struct.py:2441"""
    return _msgpack_decoder.decode(sock.recv(flags))


async def async_sock_send(sock: zmq.asyncio.Socket, obj: object) -> None:
    await sock.send(_msgpack_encoder.encode(obj))


async def async_sock_recv(sock: zmq.asyncio.Socket) -> object:
    return _msgpack_decoder.decode(await sock.recv())


# ============================================================
# 2. IPC 端点分配 —— 对应 PortArgs.init_new (server_args.py:10657):
#    用 NamedTemporaryFile 抢一个唯一路径名, zmq bind 时会把它替换成
#    自己的 unix domain socket 文件。
# ============================================================


@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str  # tokenizer 收结果的端点 (scheduler/detokenizer 都往这 PUSH)
    scheduler_input_ipc_name: str  # scheduler 收请求的端点
    detokenizer_ipc_name: str  # detokenizer 收 token id 的端点

    @staticmethod
    def init_new() -> "PortArgs":
        def ipc() -> str:
            return f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        return PortArgs(
            tokenizer_ipc_name=ipc(),
            scheduler_input_ipc_name=ipc(),
            detokenizer_ipc_name=ipc(),
        )


def get_zmq_socket(context, socket_type, endpoint: str, bind: bool):
    """对应 get_zmq_socket + config_socket (utils/network.py:233):
    HWM 设为 0 (无限) —— PUSH 永不阻塞, 对端没连上/没消费时消息堆在发送缓冲区。"""
    sock = context.socket(socket_type)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    if bind:
        sock.bind(endpoint)
    else:
        sock.connect(endpoint)
    return sock


def log(role: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{role:<11} pid={os.getpid():>6}] {msg}", flush=True)


# ============================================================
# 3. 假模型 / 假 tokenizer —— demo 里不加载真权重
# ============================================================

VOCAB = {1: "hello", 2: "world", 10: "sglang", 11: "uses", 12: "zmq", 13: "ipc", 14: "!"}
VOCAB_REV = {w: i for i, w in VOCAB.items()}
# 假 "模型": 不管输入什么, 都续写这 5 个 token (对应 5 个 decode step)
GENERATION_SCRIPT = [10, 11, 12, 13, 14]


def fake_tokenize(text: str) -> list[int]:
    return [VOCAB_REV[w] for w in text.split()]


def fake_detokenize(ids: list[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)


# ============================================================
# 4. Scheduler 子进程 —— 对应 run_scheduler_process (scheduler.py:5187)
#    + event_loop_normal (scheduler.py:1765)
# ============================================================


def run_scheduler_process(port_args: PortArgs, ready: mp.Event) -> None:
    role = "Scheduler"
    context = zmq.Context(2)  # 同 ipc_channels.py:34

    # 对应 SchedulerIpcChannels.create (ipc_channels.py:36-65):
    # scheduler 在三条通道上都是 connect 的一端 (bind 在对端进程里)。
    recv_from_tokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.scheduler_input_ipc_name, bind=False
    )
    send_to_detokenizer = get_zmq_socket(
        context, zmq.PUSH, port_args.detokenizer_ipc_name, bind=False
    )
    send_to_tokenizer = get_zmq_socket(
        context, zmq.PUSH, port_args.tokenizer_ipc_name, bind=False
    )

    waiting_queue: list[TokenizedGenerateReqInput] = []
    running: dict[str, dict] = {}  # rid -> {"step": int}
    gracefully_exit = False

    ready.set()
    log(role, "event loop started")

    # ---- event_loop_normal: recv -> process_input -> run_batch -> stream_output ----
    while not gracefully_exit:
        # (a) recv_requests: 用 NOBLOCK 把 PULL 队列一次排空
        #     对应 request_receiver.py:118-124 —— 收到 ZMQError(EAGAIN) 说明本轮没活了
        recv_reqs = []
        while True:
            try:
                recv_reqs.append(sock_recv(recv_from_tokenizer, zmq.NOBLOCK))
            except zmq.ZMQError:
                break
        if recv_reqs:
            log(role, f"<- tokenizer  drained {len(recv_reqs)} req(s): "
                      f"{[r.rid for r in recv_reqs]}")

        # (b) process_input_requests: 按消息类型分发
        for req in recv_reqs:
            if isinstance(req, TokenizedGenerateReqInput):
                waiting_queue.append(req)
            elif isinstance(req, FlushCacheReqInput):
                # 控制回包直接发给 tokenizer, 不走 detokenizer
                # 对应 scheduler.py:4120 的 send_to_tokenizer.send_output(...)
                log(role, f"flush cache {req.flush_cache_id}: replying directly to tokenizer")
                sock_send(
                    send_to_tokenizer,
                    FlushCacheReqOutput(rid=req.rid, flush_cache_id=req.flush_cache_id),
                )
            elif isinstance(req, ShutdownReq):
                gracefully_exit = True

        # (c) get_next_batch_to_run: waiting 并入 running (真实实现里还要
        #     算 prefix cache / chunked prefill / 连续批调度)
        for req in waiting_queue:
            running[req.rid] = {"step": 0}
        waiting_queue.clear()

        if running:
            # (d) run_batch: 一次前向, 每个 running 请求产出 1 个新 token
            rids, new_ids, finished = [], [], []
            for rid, st in running.items():
                tok = GENERATION_SCRIPT[st["step"]]
                st["step"] += 1
                done = st["step"] == len(GENERATION_SCRIPT)
                rids.append(rid)
                new_ids.append([tok])
                finished.append(done)

            # (e) stream_output: 只把 token id 发给 detokenizer (output_streamer.py:202-211)
            sock_send(
                send_to_detokenizer,
                BatchTokenIDOutput(rids=rids, new_token_ids=new_ids, finished=finished),
            )
            log(role, f"-> detokenizer  step: batch={len(rids)} new_ids={new_ids}")

            # 退出的请求离开 running (真实实现是 process_batch_result 里处理)
            for rid, done in zip(rids, finished):
                if done:
                    del running[rid]
        else:
            time.sleep(0.05)  # 空转, 避免 busy loop

    log(role, "gracefully_exit, bye")


# ============================================================
# 5. DetokenizerManager 子进程 —— 对应 detokenizer_manager.py:113 + event_loop:169
# ============================================================


def run_detokenizer_process(port_args: PortArgs, ready: mp.Event) -> None:
    role = "Detokenizer"
    context = zmq.Context(2)

    # 对应 detokenizer_manager.py:114-122: 自己的入口自己 bind,
    # 出口 connect 到 tokenizer 的端点。
    recv_from_scheduler = get_zmq_socket(
        context, zmq.PULL, port_args.detokenizer_ipc_name, bind=True
    )
    send_to_tokenizer = get_zmq_socket(
        context, zmq.PUSH, port_args.tokenizer_ipc_name, bind=False
    )

    ready.set()
    log(role, "event loop started")

    # 阻塞式收 —— 真实 DetokenizerManager.event_loop 就是同步阻塞 recv
    while True:
        obj = sock_recv(recv_from_scheduler)
        if isinstance(obj, BatchTokenIDOutput):
            deltas = [fake_detokenize(ids) for ids in obj.new_token_ids]
            log(role, f"-> tokenizer  {list(zip(obj.rids, deltas))}")
            sock_send(
                send_to_tokenizer,
                BatchStrOutput(rids=obj.rids, deltas=deltas, finished=obj.finished),
            )


# ============================================================
# 6. TokenizerManager (主进程, asyncio) —— 对应 tokenizer_manager.py:549 + handle_loop:2154
# ============================================================


class ReqState:
    """对应 TokenizerManager.rid_to_state 里的 state: future + 已收文本"""

    def __init__(self):
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.text_parts: list[str] = []


async def run_tokenizer_manager(port_args: PortArgs, prompts: list[str]) -> None:
    role = "Tokenizer"
    context = zmq.asyncio.Context(2)  # 异步 zmq, 同 tokenizer_manager.py:549

    # 对应 tokenizer_manager.py:550-556。注意 bind 方向:
    #   - 收结果的 PULL 在自己手里 bind (scheduler 和 detokenizer 两个 PUSH 都 connect 过来)
    #   - 发请求的 PUSH 也是自己 bind, scheduler 那边 connect
    recv_from_detokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.tokenizer_ipc_name, bind=True
    )
    send_to_scheduler = get_zmq_socket(
        context, zmq.PUSH, port_args.scheduler_input_ipc_name, bind=True
    )

    rid_to_state: dict[str, ReqState] = {}

    # ---- handle_loop: 对应 tokenizer_manager.py:2154-2167 ----
    # 单个 PULL socket 公平队列地收两个来源的消息:
    #   BatchStrOutput        <- detokenizer (生成结果)
    #   FlushCacheReqOutput   <- scheduler    (控制回包, 不经过 detokenizer)
    async def handle_loop():
        while True:
            recv_obj = await async_sock_recv(recv_from_detokenizer)
            if isinstance(recv_obj, BatchStrOutput):
                for rid, delta, done in zip(recv_obj.rids, recv_obj.deltas, recv_obj.finished):
                    state = rid_to_state.get(rid)
                    if state is None:
                        continue
                    state.text_parts.append(delta)
                    if done and not state.future.done():
                        state.future.set_result("".join(p + " " for p in state.text_parts).strip())
            elif isinstance(recv_obj, FlushCacheReqOutput):
                state = rid_to_state.pop(recv_obj.rid)
                if state is not None and not state.future.done():
                    state.future.set_result(f"flush_cache_id={recv_obj.flush_cache_id}")

    handle_task = asyncio.create_task(handle_loop())

    # ---- generate_request: 对应 tokenizer_manager.py:768 ----
    async def generate(prompt: str) -> str:
        rid = "req-" + uuid.uuid4().hex[:6]
        state = ReqState()
        rid_to_state[rid] = state
        input_ids = fake_tokenize(prompt)
        log(role, f"-> scheduler  TokenizedGenerateReqInput(rid={rid}, input_ids={input_ids})")
        await async_sock_send(
            send_to_scheduler, TokenizedGenerateReqInput(rid=rid, input_ids=input_ids)
        )
        return await state.future  # 挂起, 直到 handle_loop 按 rid 唤醒

    # 两个请求并发 —— scheduler 会把它们合成一个 batch (continuous batching)
    results = await asyncio.gather(*[generate(p) for p in prompts])
    for prompt, output in zip(prompts, results):
        log(role, f"FINISHED  prompt={prompt!r} -> output={output!r}")

    # 控制消息走同一条请求通道, 回包走 scheduler->tokenizer 直连通道
    rid = "ctrl-" + uuid.uuid4().hex[:6]
    state = ReqState()
    rid_to_state[rid] = state
    log(role, "-> scheduler  FlushCacheReqInput(flush_cache_id=42)")
    await async_sock_send(
        send_to_scheduler, FlushCacheReqInput(rid=rid, flush_cache_id=42)
    )
    log(role, f"FINISHED  flush_cache -> {await state.future}")

    # 关停 scheduler (真实 sglang 是 SIGINT 信号驱动 gracefully_exit)
    await async_sock_send(send_to_scheduler, ShutdownReq(rid="shutdown"))
    await asyncio.sleep(0.2)
    handle_task.cancel()


# ============================================================
# 7. 主入口 —— 对应 Engine._launch_subprocesses (entrypoints/engine.py:1022):
#    先起 scheduler, 再起 detokenizer, 最后在主进程里建 TokenizerManager。
#    起进程顺序无所谓: zmq connect 方先发消息只会堆在本地发送缓冲 (HWM=0),
#    对端 bind 好之后自动送达。
# ============================================================


def main() -> None:
    port_args = PortArgs.init_new()
    log("Main", f"ipc endpoints: {port_args}")

    sched_ready, detok_ready = mp.Event(), mp.Event()
    sched = mp.Process(target=run_scheduler_process, args=(port_args, sched_ready))
    detok = mp.Process(target=run_detokenizer_process, args=(port_args, detok_ready))
    sched.start()
    detok.start()
    sched_ready.wait()
    detok_ready.wait()

    try:
        asyncio.run(run_tokenizer_manager(port_args, ["hello world", "hello world"]))
    finally:
        sched.join(timeout=5)
        # 对应 Engine.shutdown 里的 kill_process_tree: detokenizer 是阻塞 recv,
        # 由主进程直接终结
        if detok.is_alive():
            detok.terminate()
        detok.join(timeout=5)
        for path in (
            port_args.tokenizer_ipc_name,
            port_args.scheduler_input_ipc_name,
            port_args.detokenizer_ipc_name,
        ):
            try:
                os.unlink(path.removeprefix("ipc://"))
            except OSError:
                pass
        log("Main", "all processes exited, cleaned up ipc files")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # socket 不能跨 fork 继承, spawn 最稳
    main()
