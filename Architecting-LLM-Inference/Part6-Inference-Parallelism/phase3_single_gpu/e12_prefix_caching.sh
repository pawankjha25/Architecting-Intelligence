#!/usr/bin/env bash
# E12 — Prefix Caching
# ======================
# Goal:   Measure TTFT reduction from shared prefix reuse.
# Run on: RunPod RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e12_prefix_caching"
mkdir -p "$RESULTS_DIR"

echo "=== E12: Prefix Caching ==="

run_benchmark() {
  local PREFIX_CACHING=$1
  local LABEL=$2
  local EXTRA_ARGS=""
  [[ "$PREFIX_CACHING" == "1" ]] && EXTRA_ARGS="--enable-prefix-caching"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 8192 \
    --dtype float16 \
    --port "$PORT" \
    $EXTRA_ARGS \
    --disable-log-requests &
  local PID=$!

  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

  # Workload: same long system prompt + varied user queries (high cache hit rate)
  python phase3_single_gpu/e12_prefix_caching_client.py \
    --port "$PORT" \
    --model "$MODEL" \
    --label "$LABEL" \
    --output-dir "$RESULTS_DIR"

  kill "$PID" 2>/dev/null || true
  sleep 3
}

run_benchmark 0 "no_prefix_cache"
run_benchmark 1 "with_prefix_cache"

echo "Done. Results → $RESULTS_DIR/"
echo "Key metric: TTFT reduction % for requests with shared system prompt."
