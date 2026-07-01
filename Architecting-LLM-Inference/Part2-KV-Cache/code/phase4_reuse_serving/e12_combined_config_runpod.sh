#!/bin/bash
# E12 (step 2 of 2) -- Combined Optimization Configuration
# Run on: RunPod RTX 3090 (or better), single GPU.
#
# IMPORTANT: run `python3 phase4_reuse_serving/e12_compat_matrix.py` first
# and confirm your vLLM version supports every flag used below. Do not
# assume they all compose cleanly without checking.
#
# Baseline:  bf16, contiguous-style defaults, no prefix reuse, GPU-only.
# Optimized: paged allocation (block_size=16) + prefix caching +
#            fp8 KV cache quantization + cache-aware admission via
#            gpu-memory-utilization tuning.
# Sliding-window eviction is intentionally NOT stacked here -- it's only
# safe for models that natively use it (see e12_compat_matrix.py). Use
# Mistral instead of Qwen if you want to add that axis, and measure it
# as its own variable, not folded silently into "optimized."
#
# Usage: bash e12_combined_config_runpod.sh

set -e
MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e12_combined"
mkdir -p "$RESULTS_DIR"

echo "=== BASELINE: no prefix caching, bf16, gpu-only ==="
vllm serve "$MODEL" \
  --block-size 16 \
  --gpu-memory-utilization 0.85 \
  --port 8000 &
SERVER_PID=$!
until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done

python3 phase2_allocation/e05_client.py \
  --port 8000 --block-size 16 --concurrency 64 --num-requests 300 \
  --output "$RESULTS_DIR/baseline.json" || true

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
sleep 5

echo "=== OPTIMIZED: paged + prefix caching + fp8 kv cache ==="
vllm serve "$MODEL" \
  --block-size 16 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --port 8000 &
SERVER_PID=$!
until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done

python3 phase2_allocation/e05_client.py \
  --port 8000 --block-size 16 --concurrency 64 --num-requests 300 \
  --output "$RESULTS_DIR/optimized.json" || true

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Done. Compare $RESULTS_DIR/baseline.json vs optimized.json --"
echo "expect gains in max concurrency and TTFT (prefix hits), and a small"
echo "quality question mark from fp8 kv cache that E09 lets you reason about"
echo "independently of this run."
