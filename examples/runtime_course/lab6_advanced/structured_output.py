"""
Lab 6: Structured output with a JSON schema constraint.

Prerequisite: launch a server first, e.g.
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

Usage:
    python structured_output.py --port 30000
"""

import argparse
import json

import requests

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
    "required": ["name", "population", "landmarks"],
}

PROMPT = "Give me information about the capital of France in JSON format."


def generate(base_url: str, json_schema=None):
    sampling_params = {"max_new_tokens": 128, "temperature": 0}
    if json_schema is not None:
        sampling_params["json_schema"] = json.dumps(json_schema)
    response = requests.post(
        f"{base_url}/generate",
        json={"text": PROMPT, "sampling_params": sampling_params},
    )
    response.raise_for_status()
    return response.json()["text"]


def main(base_url: str):
    print("Without json_schema constraint:")
    print(generate(base_url))
    print()

    print("With json_schema constraint:")
    constrained = generate(base_url, JSON_SCHEMA)
    print(constrained)
    print()

    # Verify the constrained output is valid JSON that satisfies the schema keys.
    parsed = json.loads(constrained)
    missing = [key for key in JSON_SCHEMA["required"] if key not in parsed]
    if missing:
        print(f"FAILED: missing required keys: {missing}")
    else:
        print("OK: constrained output is valid JSON with all required keys.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    main(f"http://{args.host}:{args.port}")
