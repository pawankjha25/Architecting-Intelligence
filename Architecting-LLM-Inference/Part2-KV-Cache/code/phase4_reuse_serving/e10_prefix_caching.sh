#!/bin/bash
# E10 -- Prefix Caching Cold vs. Warm
# Run on: RunPod RTX 3090 (or better), single GPU.
#
# Measures TTFT and prefill-tokens-avoided when a long shared prefix
# (e.g. a system prompt or RAG document) is reused across many requests,
# with vLLM's automatic prefix caching enabled vs. disabled.
#
# Usage: bash e10_prefix_caching.sh

set -e
MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e10_prefix_caching"
mkdir -p "$RESULTS_DIR"

for MODE in disabled enabled; do
  echo "=== prefix_caching=$MODE ==="

  FLAG=""
  if [ "$MODE" == "enabled" ]; then
    FLAG="--enable-prefix-caching"
  fi

  vllm serve "$MODEL" $FLAG --gpu-memory-utilization 0.85 --port 8000 &
  SERVER_PID=$!
  until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done

  python3 phase4_reuse_serving/e10_client.py \
    --port 8000 \
    --mode "$MODE" \
    --shared-prefix-len 2000 \
    --num-requests 100 \
    --output "$RESULTS_DIR/${MODE}.json" || true

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 5
done

echo "Done. Compare $RESULTS_DIR/disabled.json vs $RESULTS_DIR/enabled.json -- look at ttft_ms_mean."
