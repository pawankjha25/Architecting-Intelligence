#!/usr/bin/env bash
# E14 — KV Cache CPU Offloading
# ================================
# Goal:   Show how --cpu-offload-gb extends effective KV capacity
#         at the cost of throughput (PCIe bandwidth bottleneck).
# Run on: RunPod RTX 3090 (24GB GPU + CPU RAM)
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e14_kv_offloading"
mkdir -p "$RESULTS_DIR"

echo "=== E14: KV Cache CPU Offloading ==="

run_with_offload() {
  local CPU_OFFLOAD_GB=$1
  local EXTRA_ARGS=""
  [[ "$CPU_OFFLOAD_GB" -gt 0 ]] && EXTRA_ARGS="--cpu-offload-gb $CPU_OFFLOAD_GB"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 16384 --dtype float16 \
    --port "$PORT" $EXTRA_ARGS --disable-log-requests &
  local PID=$!
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

  # Record GPU memory before benchmark
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    > "$RESULTS_DIR/gpu_mem_offload${CPU_OFFLOAD_GB}.txt"

  python -m vllm.benchmarks.benchmark_serving \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name sharegpt --num-prompts 100 --request-rate 5 \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "offload_${CPU_OFFLOAD_GB}gb.json"

  kill "$PID" 2>/dev/null || true; sleep 5
}

for OFFLOAD_GB in 0 4 8 16; do
  echo "  cpu_offload_gb=$OFFLOAD_GB..."
  run_with_offload "$OFFLOAD_GB"
done

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: max sequence length supported, throughput penalty vs offload GB."
