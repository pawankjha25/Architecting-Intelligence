#!/bin/bash
# E11 (Phase B) -- Real vLLM CPU-offload comparison.
# Run on: RunPod RTX 3090 (or better), single GPU.
#
# Sweeps vLLM's --swap-space (CPU offload capacity in GB) and measures
# throughput/latency penalty vs. a GPU-only baseline (--swap-space 0).
# This is the real-engine counterpart to e11_offload_prototype.py's
# manual round-trip measurement -- expect a much smaller penalty here
# since vLLM overlaps transfer with compute.
#
# Usage: bash e11_cpu_offload_runpod.sh

set -e
MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e11_cpu_offload"
mkdir -p "$RESULTS_DIR"

for SWAP_GB in 0 4 8 16; do
  echo "=== swap_space=${SWAP_GB}GB ==="

  vllm serve "$MODEL" \
    --swap-space "$SWAP_GB" \
    --gpu-memory-utilization 0.85 \
    --port 8000 &
  SERVER_PID=$!
  until curl -s http://localhost:8000/health > /dev/null; do sleep 2; done

  python3 phase2_allocation/e05_client.py \
    --port 8000 \
    --block-size 16 \
    --concurrency 64 \
    --num-requests 300 \
    --output "$RESULTS_DIR/swap_${SWAP_GB}gb.json" || true

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 5
done

echo "Done. Compare throughput/TTFT across $RESULTS_DIR/swap_*.json -- expect a small"
echo "throughput dip as swap_space increases, in exchange for surviving memory spikes"
echo "that would otherwise OOM-reject requests at swap_space=0."
