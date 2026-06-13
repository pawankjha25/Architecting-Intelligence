#!/usr/bin/env bash
# E27 — CUDA Graphs: On vs Off
# ==============================
# Goal:   Measure the decode throughput and latency gain from CUDA Graph capture.
#         vLLM enables CUDA Graphs by default for the decode phase.
#         --enforce-eager disables them, forcing eager (step-by-step) execution.
#
# What CUDA Graphs do:
#   - vLLM records the decode forward pass as a CUDA Graph during warmup.
#   - Subsequent decode steps replay the graph via a single CUDA API call.
#   - This eliminates Python/CUDA launch overhead on every decode step.
#   - Effect is strongest for small batch sizes (low concurrency) where
#     kernel launch overhead is a large fraction of per-step time.
#   - At very high batch sizes the gain shrinks (compute dominates overhead).
#
# Run on: RunPod RTX 3090 (single GPU)
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e27_cuda_graphs"
mkdir -p "$RESULTS_DIR"

echo "=== E27: CUDA Graphs On vs Off ==="
echo "Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

run_benchmark() {
  local LABEL=$1
  local EAGER_FLAG=$2   # "" or "--enforce-eager"

  echo "--- $LABEL ($EAGER_FLAG) ---"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 4096 \
    --dtype float16 \
    --port "$PORT" \
    $EAGER_FLAG \
    --disable-log-requests &
  local PID=$!

  echo "  Waiting for server (graph capture may take ~30s)..."
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
  echo "  Server ready."

  # Note server startup time — CUDA Graph capture happens during warmup
  # The "ready" signal comes AFTER capture, so startup delta is capture time.

  # Sweep: low concurrency where CUDA Graph benefit is largest
  for RPS in 1 2 4 8 16 32 64; do
    echo "  request_rate=$RPS..."
    python -m vllm.benchmarks.benchmark_serving \
      --backend openai \
      --base-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --num-prompts 200 \
      --request-rate "$RPS" \
      --percentile-metrics "ttft,tpot,e2e_latency" \
      --save-result \
      --result-dir "$RESULTS_DIR" \
      --result-filename "${LABEL}_rps${RPS}.json"
  done

  kill "$PID" 2>/dev/null || true
  sleep 5
}

# ── 1. CUDA Graphs enabled (vLLM default) ─────────────────────────────────────
run_benchmark "cuda_graphs_on"  ""

# ── 2. Eager mode (CUDA Graphs disabled) ──────────────────────────────────────
run_benchmark "cuda_graphs_off" "--enforce-eager"

# ── 3. Print comparison ───────────────────────────────────────────────────────
echo ""
echo "=== Results summary ==="
python phase3_single_gpu/e27_cuda_graphs_compare.py --results-dir "$RESULTS_DIR"

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo ""
echo "Expected pattern:"
echo "  Low RPS  (1–4):   CUDA Graphs give largest speedup (10–30% TPOT reduction)"
echo "  Mid RPS  (8–16):  Moderate speedup"
echo "  High RPS (32+):   Gain diminishes — compute dominates launch overhead"
echo ""
echo "Startup time delta = CUDA Graph capture time (~20–60s for 7B model)."
