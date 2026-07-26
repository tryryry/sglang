"""
Lab 3: Measure the effect of RadixAttention prefix caching on TTFT.

Prerequisite: launch a server first, e.g.
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

Usage:
    python prefix_cache_benchmark.py --port 30000
"""

import argparse
import json
import time

import requests

# A long shared system prompt (~1000 tokens) so the prefix cache effect is visible.
LONG_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and precise assistant. "
    "Always answer carefully and concisely. "
) * 60


def send_generate(base_url: str, text: str, max_new_tokens: int = 32):
    """Send a streaming /generate request and measure TTFT and cached tokens."""
    tic = time.perf_counter()
    ttft = None
    meta_info = {}
    with requests.post(
        f"{base_url}/generate",
        json={
            "text": text,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            "stream": True,
        },
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=False):
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - tic
            meta_info = json.loads(payload).get("meta_info", {})
    return ttft, meta_info.get("cached_tokens", 0)


def flush_cache(base_url: str):
    response = requests.post(f"{base_url}/flush_cache")
    print(f"/flush_cache -> {response.status_code}")


def main(base_url: str):
    results = []

    prompt = LONG_SYSTEM_PROMPT + "\nUser: What is the capital of France?\nAssistant:"

    # 1. Cold request: no cache.
    ttft, cached = send_generate(base_url, prompt)
    results.append(("1st request (cold)", cached, ttft))

    # 2. Same prefix again: should hit the radix cache (if enabled).
    ttft, cached = send_generate(base_url, prompt)
    results.append(("2nd request (same prefix)", cached, ttft))

    # 3. Flush cache and resend: hit rate should drop to zero.
    flush_cache(base_url)
    time.sleep(1)
    ttft, cached = send_generate(base_url, prompt)
    results.append(("after /flush_cache", cached, ttft))

    # 4. Multi-turn conversation: each turn extends the shared prefix.
    print("\nMulti-turn conversation:")
    history = LONG_SYSTEM_PROMPT
    for turn in range(1, 6):
        history += f"\nUser: Tell me fact #{turn} about the ocean.\nAssistant:"
        ttft, cached = send_generate(base_url, history)
        print(f"  turn {turn}: cached_tokens={cached}, TTFT={ttft:.3f}s")
        history += " (answer omitted)"

    print("\nSummary:")
    print(f"{'scenario':<30}{'cached_tokens':>15}{'TTFT (s)':>12}")
    for name, cached, ttft in results:
        print(f"{name:<30}{cached:>15}{ttft:>12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    main(f"http://{args.host}:{args.port}")
