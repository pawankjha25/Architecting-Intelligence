#!/usr/bin/env bash
# E17 — Data Parallelism (Multi-Replica)
# =========================================
# Goal:   Show linear throughput scaling with 2 replicas, latency unchanged.
#         This is the correct pattern when the model fits on 1 GPU.
# Run on: RunPod 2× RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct (fits on 1× RTX 3090 in fp16)
#
# Architecture:
#   GPU 0 → vLLM replica 0 (port 8000)
#   GPU 1 → vLLM replica 1 (port 8001)
#   Simple round-robin load balancer (nginx or haproxy)

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e17_data_parallel"
mkdir -p "$RESULTS_DIR"

echo "=== E17: Data Parallelism (2 replicas) ==="
echo "Model: $MODEL"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ── Start replica 0 on GPU 0 ──────────────────────────────────────────────────
echo "Starting replica 0 on GPU 0 (port 8000)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --port 8000 --disable-log-requests &
PID0=$!

# ── Start replica 1 on GPU 1 ──────────────────────────────────────────────────
echo "Starting replica 1 on GPU 1 (port 8001)..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --port 8001 --disable-log-requests &
PID1=$!

echo "Waiting for both replicas..."
until curl -sf "http://localhost:8000/health" > /dev/null 2>&1; do sleep 2; done
until curl -sf "http://localhost:8001/health" > /dev/null 2>&1; do sleep 2; done
echo "Both replicas ready."

# ── Baseline: single replica ─────────────────────────────────────────────────
echo ""
echo "[1/2] Benchmarking single replica (port 8000)..."
python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:8000" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 400 --request-rate 20 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "single_replica.json"

# ── Round-robin across 2 replicas ────────────────────────────────────────────
echo ""
echo "[2/2] Benchmarking 2-replica setup (round-robin)..."
python phase4_parallelism/e17_roundrobin_client.py \
  --ports 8000 8001 \
  --model "$MODEL" \
  --num-requests 400 \
  --request-rate 40 \
  --output "$RESULTS_DIR/two_replicas.json"

kill "$PID0" "$PID1" 2>/dev/null || true

echo "Done. Results → $RESULTS_DIR/"
echo "Expected: throughput doubles, latency stays flat."
echo "This is the best scaling strategy when model fits on one GPU."
