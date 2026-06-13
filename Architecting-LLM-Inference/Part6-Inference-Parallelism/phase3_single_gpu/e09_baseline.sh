#!/usr/bin/env bash
# E09 — vLLM Baseline
# =====================
# Goal:   Establish real GPU baseline. Measure TTFT, TPOT, throughput.
# Run on: RunPod RTX 3090 (single GPU)
# Model:  Llama 3.1 8B Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"   # open, Apache 2.0, no gating
# Alt (requires HuggingFace access): meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=8192
PORT=8000
RESULTS_DIR="results/e09_baseline"
mkdir -p "$RESULTS_DIR"

echo "=== E09: vLLM Baseline === "
echo "Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ── Step 1: Start vLLM server ──────────────────────────────────────────────────
echo "[1/3] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype float16 \
  --port "$PORT" \
  --disable-log-requests &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  Server ready."
    break
  fi
  sleep 2
done

# ── Step 2: Run benchmark ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Running benchmark (ShareGPT-style workload)..."

# vLLM's built-in benchmark
python -m vllm.benchmarks.benchmark_serving \
  --backend openai \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --dataset-name sharegpt \
  --num-prompts 200 \
  --request-rate 10 \
  --save-result \
  --result-dir "$RESULTS_DIR" \
  --result-filename "baseline_rps10.json"

# Vary request rate
for RPS in 1 5 10 20 50; do
  echo "  Testing request_rate=$RPS req/s..."
  python -m vllm.benchmarks.benchmark_serving \
    --backend openai \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --num-prompts 100 \
    --request-rate "$RPS" \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "baseline_rps${RPS}.json"
done

# ── Step 3: GPU memory snapshot ───────────────────────────────────────────────
echo ""
echo "[3/3] GPU memory snapshot:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu \
           --format=csv

kill "$SERVER_PID" 2>/dev/null || true
echo ""
echo "Done. Results in $RESULTS_DIR/"
