"""
Lab 2: Offline batch inference of 100 prompts with throughput statistics.

Usage:
    python batch_inference_100.py --model-path Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import dataclasses
import time

import sglang as sgl
from sglang.srt.server_args import ServerArgs

TOPICS = [
    "the history of the Roman Empire",
    "how photosynthesis works",
    "the rules of chess",
    "the theory of relativity",
    "the water cycle",
    "how vaccines work",
    "the French Revolution",
    "machine learning basics",
    "the structure of DNA",
    "plate tectonics",
]


def build_prompts(num_prompts: int):
    return [
        f"Question {i}: Briefly explain {TOPICS[i % len(TOPICS)]} in two sentences.\nAnswer:"
        for i in range(num_prompts)
    ]


def main(server_args: ServerArgs, num_prompts: int):
    prompts = build_prompts(num_prompts)
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 64,
    }

    llm = sgl.Engine(**dataclasses.asdict(server_args))

    tic = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - tic

    # Show a few sample outputs.
    for prompt, output in zip(prompts[:3], outputs[:3]):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    total_output_tokens = sum(
        output["meta_info"]["completion_tokens"] for output in outputs
    )
    print("===============================")
    print(f"Processed {len(prompts)} prompts in {elapsed:.2f}s")
    print(f"Throughput: {len(prompts) / elapsed:.2f} requests/s")
    print(f"Output token throughput: {total_output_tokens / elapsed:.2f} tokens/s")

    llm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=100)
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    num_prompts = args.num_prompts
    server_args = ServerArgs.from_cli_args(args)
    main(server_args, num_prompts)
