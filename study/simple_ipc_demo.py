"""Minimal three-process ZMQ IPC demo.

Message flow:
    Tokenizer -> Scheduler -> Detokenizer -> Tokenizer

All three roles run in separate ``multiprocessing.Process`` instances. The
parent process only creates IPC endpoints and manages process lifetimes.
"""

import multiprocessing as mp
import os
import tempfile

import zmq


def log(role: str, message: str) -> None:
    print(f"[{role:<11} pid={os.getpid()}] {message}", flush=True)


def run_tokenizer(request_endpoint: str, result_endpoint: str) -> None:
    context = zmq.Context()
    send_request = context.socket(zmq.PUSH)
    receive_result = context.socket(zmq.PULL)
    send_request.bind(request_endpoint)
    receive_result.bind(result_endpoint)

    request = {"request_id": "req-1", "prompt": "hello ipc"}
    log("Tokenizer", f"send request: {request}")
    send_request.send_json(request)

    result = receive_result.recv_json()
    log("Tokenizer", f"receive result: {result}")

    send_request.close()
    receive_result.close()
    context.term()


def run_scheduler(request_endpoint: str, token_endpoint: str) -> None:
    context = zmq.Context()
    receive_request = context.socket(zmq.PULL)
    send_tokens = context.socket(zmq.PUSH)
    receive_request.connect(request_endpoint)
    send_tokens.connect(token_endpoint)

    request = receive_request.recv_json()
    log("Scheduler", f"receive request: {request}")

    token_message = {
        "request_id": request["request_id"],
        "token_ids": [73, 80, 67],
    }
    log("Scheduler", f"send tokens: {token_message}")
    send_tokens.send_json(token_message)

    receive_request.close()
    send_tokens.close()
    context.term()


def run_detokenizer(token_endpoint: str, result_endpoint: str) -> None:
    context = zmq.Context()
    receive_tokens = context.socket(zmq.PULL)
    send_result = context.socket(zmq.PUSH)
    receive_tokens.bind(token_endpoint)
    send_result.connect(result_endpoint)

    token_message = receive_tokens.recv_json()
    log("Detokenizer", f"receive tokens: {token_message}")

    result = {
        "request_id": token_message["request_id"],
        "text": "".join(chr(token_id) for token_id in token_message["token_ids"]),
    }
    log("Detokenizer", f"send result: {result}")
    send_result.send_json(result)

    receive_tokens.close()
    send_result.close()
    context.term()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sglang-ipc-demo-") as ipc_dir:
        request_endpoint = f"ipc://{ipc_dir}/requests.sock"
        token_endpoint = f"ipc://{ipc_dir}/tokens.sock"
        result_endpoint = f"ipc://{ipc_dir}/results.sock"

        process_context = mp.get_context("spawn")
        processes = [
            process_context.Process(
                name="tokenizer",
                target=run_tokenizer,
                args=(request_endpoint, result_endpoint),
            ),
            process_context.Process(
                name="scheduler",
                target=run_scheduler,
                args=(request_endpoint, token_endpoint),
            ),
            process_context.Process(
                name="detokenizer",
                target=run_detokenizer,
                args=(token_endpoint, result_endpoint),
            ),
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        failed = [process.name for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"Processes failed: {failed}")


if __name__ == "__main__":
    main()