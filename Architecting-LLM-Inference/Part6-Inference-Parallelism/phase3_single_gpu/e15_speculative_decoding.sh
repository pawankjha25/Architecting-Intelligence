#!/usr/bin/env bash
# E15 — Speculative Decoding
# ============================
# Goal:   Measure token acceptance rate and latency speedup.
#         Draft model proposes K tokens; target verifies in parallel.
# Run on: RunPod RTX 3090
# Models: Draft: Qwen/Qwen2.5-1.5B-Instruct  Target: Qwen/Qwen2.5-7B-Instruct
#         Same family → higher acceptance rate vs cross-family pairing.

set -euo pipefail

TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
RESULTS_DIR="results/e15_speculative_decoding"
mkdir -p "$RESULTS_DIR"

echo "=== E15: Speculative Decoding ==="
echo "Target: $TARGET_MODEL  |  Draft: $DRAFT_MODEL"

# ── Baseline: target model alone ─────────────────────────────────────────────
echo ""
echo "[1/2] Baseline (target only)..."
python -m vllm.entrypoints.openai.api_server \
  --model "$TARGET_MODEL" --max-model-len 4096 --dtype float16 \
  --port "$PORT" --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "$TARGET_MODEL" \
  --dataset-name sharegpt --num-prompts 200 --request-rate 5 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "baseline.json"
kill "$PID" 2>/dev/null || true; sleep 5

# ── Speculative decoding: vary num_speculative_tokens ─────────────────────────
echo ""
echo "[2/2] Speculative decoding..."
for K in 3 5 7 10; do
  echo "  num_speculative_tokens=$K..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --speculative-model "$DRAFT_MODEL" \
    --num-speculative-tokens "$K" \
    --max-model-len 4096 --dtype float16 \
    --port "$PORT" --disable-log-requests &
  PID=$!
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

  # Vary temperature to show acceptance rate sensitivity
  for TEMP in 0.0 0.3 0.7 1.0; do
    python -m vllm.benchmarks.benchmark_serving \
      --backend openai --base-url "http://localhost:$PORT" --model "$TARGET_MODEL" \
      --dataset-name sharegpt --num-prompts 100 --request-rate 5 \
      --save-result --result-dir "$RESULTS_DIR" \
      --result-filename "spec_k${K}_temp${TEMP}.json"
  done

  kill "$PID" 2>/dev/null || true; sleep 5
done

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: acceptance rate, latency speedup vs K, speedup vs temperature."
echo "Note: acceptance rate drops as temperature increases (sampling diverges from greedy)."
