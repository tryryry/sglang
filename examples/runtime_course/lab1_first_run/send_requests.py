"""
Lab 1: Send requests to a running SGLang server in three ways.

Prerequisite: launch a server first, e.g.
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

Usage:
    python send_requests.py --port 30000
"""

import argparse
import json
import time

import requests


def native_api(base_url: str):
    print("=" * 40)
    print("1. Native /generate API")
    print("=" * 40)
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0},
        },
    )
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def openai_api(base_url: str):
    print("=" * 40)
    print("2. OpenAI-compatible /v1/chat/completions API")
    print("=" * 40)
    import openai

    client = openai.Client(base_url=f"{base_url}/v1", api_key="None")
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
        temperature=0,
        max_tokens=64,
    )
    print(response.choices[0].message.content)


def streaming_vs_non_streaming(base_url: str):
    print("=" * 40)
    print("3. Streaming vs non-streaming")
    print("=" * 40)
    import openai

    client = openai.Client(base_url=f"{base_url}/v1", api_key="None")
    prompt = "Write a short poem about the moon."

    # Non-streaming: single response after generation finishes.
    tic = time.perf_counter()
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
    )
    total = time.perf_counter() - tic
    print(f"[non-streaming] total latency: {total:.3f}s")
    print(response.choices[0].message.content)
    print()

    # Streaming: chunks arrive as tokens are generated.
    tic = time.perf_counter()
    first_token_time = None
    stream = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        stream=True,
    )
    print("[streaming] output: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter() - tic
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    print(f"[streaming] time to first token (TTFT): {first_token_time:.3f}s")


def admin_apis(base_url: str):
    print("=" * 40)
    print("4. Admin APIs")
    print("=" * 40)
    health = requests.get(f"{base_url}/health")
    print(f"/health -> {health.status_code}")
    info = requests.get(f"{base_url}/get_server_info").json()
    keys_of_interest = ["model_path", "tp_size", "max_running_requests", "version"]
    for key in keys_of_interest:
        if key in info:
            print(f"/get_server_info {key} = {info[key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    native_api(base_url)
    openai_api(base_url)
    streaming_vs_non_streaming(base_url)
    admin_apis(base_url)
