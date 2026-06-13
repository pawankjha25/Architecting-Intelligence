#!/usr/bin/env bash
# E19 — Pipeline Parallelism PP=2
# ==================================
# Goal:   Split model by layer groups across 2 GPUs.
#         Show pipeline bubble effect — TTFT increases, throughput can improve
#         for large batches once the pipeline is filled.
#
# Model:  Qwen/Qwen2.5-14B-Instruct (same as E18 for fair comparison)
# Run on: RunPod 2× RTX 3090

set -euo pipefail

MODEL="Qwen/Qwen2.5-14B-Instruct"
PORT=8000
RESULTS_DIR="results/e19_pipeline_parallel"
mkdir -p "$RESULTS_DIR"

echo "=== E19: Pipeline Parallelism PP=2 ==="
echo "Model: $MODEL"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --pipeline-parallel-size 2 \
  --port "$PORT" \
  --disable-log-requests &
PID=$!

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
echo "Server ready (PP=2)."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for RPS in 1 5 10 20; do
  python -m vllm.benchmarks.benchmark_serving \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name sharegpt --num-prompts 200 --request-rate "$RPS" \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "pp2_rps${RPS}.json"
done

kill "$PID" 2>/dev/null || true

echo "Done. Results → $RESULTS_DIR/"
echo "Compare with E18 (TP=2):"
echo "  PP: higher TTFT (pipeline bubble), better memory balance for deep models"
echo "  TP: lower TTFT (no bubble), more communication overhead"
