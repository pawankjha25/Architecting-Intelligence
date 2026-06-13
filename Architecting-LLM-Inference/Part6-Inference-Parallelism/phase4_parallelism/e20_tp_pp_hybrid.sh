#!/usr/bin/env bash
# E20 — TP+PP Hybrid
# ====================
# Goal:   Combine tensor and pipeline parallelism. Requires 4 GPUs ideally.
#         On 2 GPUs: demonstrate TP=2,PP=1 vs TP=1,PP=2 tradeoff comparison.
#
# Model:  Qwen/Qwen2.5-14B-Instruct
# Run on: RunPod 2× RTX 3090 (limited — shows TP=2+PP=1, not full hybrid)
#         For full TP=2+PP=2 hybrid you need 4 GPUs.

set -euo pipefail

MODEL="Qwen/Qwen2.5-14B-Instruct"
PORT=8000
RESULTS_DIR="results/e20_tp_pp_hybrid"
mkdir -p "$RESULTS_DIR"

echo "=== E20: TP+PP Hybrid ==="
echo "Model: $MODEL | Hardware: 2× RTX 3090"
echo "Note: true TP+PP hybrid (TP=2, PP=2) needs 4 GPUs."
echo "      This experiment compares TP=2 vs PP=2 on 2 GPUs as the hybrid baseline.\n"

# ── Config 1: TP=2, PP=1 (tensor parallel only — same as E18) ─────────────────
echo "[1/2] TP=2, PP=1..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --port "$PORT" --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 200 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "tp2_pp1.json"
kill "$PID" 2>/dev/null || true; sleep 5

# ── Config 2: TP=1, PP=2 (pipeline parallel only — same as E19) ───────────────
echo "[2/2] TP=1, PP=2..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --port "$PORT" --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 200 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "tp1_pp2.json"
kill "$PID" 2>/dev/null || true

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo ""
echo "Article takeaway:"
echo "  On 2 GPUs: TP=2 usually wins for small models (lower TTFT, less bubble)."
echo "  PP=2 wins for very deep models with few layers per stage."
echo "  True hybrid (TP=2+PP=2) needs 4 GPUs and is used for 70B+ models."
