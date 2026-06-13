#!/usr/bin/env bash
# E11 — Continuous Batching Deep Dive
# =====================================
# Goal:   Measure real throughput gains over static batching under load.
# Run on: RunPod RTX 3090 (single GPU)
# Model:  mistralai/Mistral-7B-Instruct-v0.3  (open, no gating)
# Alt:    Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # open, Apache 2.0
PORT=8000
RESULTS_DIR="results/e11_continuous_batching"
mkdir -p "$RESULTS_DIR"

echo "=== E11: Continuous Batching ==="
echo "Model: $MODEL"

# ── Start server with continuous batching (default in vLLM) ───────────────────
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 8192 \
  --dtype float16 \
  --port "$PORT" \
  --max-num-seqs 256 \
  --disable-log-requests &
SERVER_PID=$!

echo "Waiting for server..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
echo "Server ready."

# ── Sweep concurrency levels ───────────────────────────────────────────────────
for RPS in 1 5 10 20 50 100; do
  echo "  request_rate=$RPS req/s..."
  python -m vllm.benchmarks.benchmark_serving \
    --backend openai \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --num-prompts 200 \
    --request-rate "$RPS" \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "cb_rps${RPS}.json"
done

kill "$SERVER_PID" 2>/dev/null || true
echo "Done. Results → $RESULTS_DIR/"
