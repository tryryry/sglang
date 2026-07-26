#!/bin/bash
# Lab 4: Sweep scheduler-related server parameters and benchmark each configuration.
#
# Usage:
#   bash bench_params_sweep.sh [MODEL_PATH] [PORT]
#
# Example:
#   bash bench_params_sweep.sh Qwen/Qwen2.5-0.5B-Instruct 30000

set -e

MODEL_PATH=${1:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${2:-30000}
NUM_PROMPTS=${NUM_PROMPTS:-200}
RESULT_DIR=$(mktemp -d /tmp/sglang_lab4.XXXXXX)

CONFIGS=(
    "baseline|"
    "max_running_32|--max-running-requests 32"
    "max_running_256|--max-running-requests 256"
    "chunked_prefill_512|--chunked-prefill-size 512"
    "chunked_prefill_8192|--chunked-prefill-size 8192"
    "mem_frac_0.5|--mem-fraction-static 0.5"
    "mem_frac_0.85|--mem-fraction-static 0.85"
)

wait_for_server() {
    for _ in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null; then
            return 0
        fi
        sleep 5
    done
    echo "Server failed to start" >&2
    return 1
}

for config in "${CONFIGS[@]}"; do
    name="${config%%|*}"
    extra_args="${config#*|}"
    echo "===================================================="
    echo "Config: ${name} (${extra_args:-default})"
    echo "===================================================="

    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --port "${PORT}" \
        ${extra_args} \
        > "${RESULT_DIR}/server_${name}.log" 2>&1 &
    SERVER_PID=$!

    wait_for_server

    python -m sglang.bench_serving \
        --backend sglang \
        --port "${PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        | tee "${RESULT_DIR}/bench_${name}.log"

    kill "${SERVER_PID}"
    wait "${SERVER_PID}" 2> /dev/null || true
    sleep 5
done

echo ""
echo "All results saved in ${RESULT_DIR}"
echo "Summary (grep key metrics):"
grep -H -E "Request throughput|Output token throughput|Mean TTFT|Mean ITL" "${RESULT_DIR}"/bench_*.log || true
