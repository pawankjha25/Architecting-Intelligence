#!/bin/bash
# E05 (Phase B) -- Real block-size sweep against a live vLLM server.
# Run on: RunPod RTX 3090 (or better), single GPU.
#
# CAUTION: supported block sizes are runtime/kernel-version dependent.
# Recent vLLM releases require block_size >= 16 on most backends.
# Check `vllm serve --help` for --block-size on YOUR installed version
# before assuming 8 works.
#
# Usage: bash e05_block_size_sweep_runpod.sh

set -e
MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e05_block_size_sweep"
mkdir -p "$RESULTS_DIR"

for BLOCK_SIZE in 16 32 64 128; do
  echo "=== block_size=$BLOCK_SIZE ==="

  vllm serve "$MODEL" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization 0.85 \
    --port 8000 &
  SERVER_PID=$!

  # wait for server to come up
  until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done

  python3 phase2_allocation/e05_client.py \
    --port 8000 \
    --block-size "$BLOCK_SIZE" \
    --concurrency 32 \
    --num-requests 200 \
    --output "$RESULTS_DIR/block_${BLOCK_SIZE}.json" || true

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 5
done

echo "Done. Results in $RESULTS_DIR/"
