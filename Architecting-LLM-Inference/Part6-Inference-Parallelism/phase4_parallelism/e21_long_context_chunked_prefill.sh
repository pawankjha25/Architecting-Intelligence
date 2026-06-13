#!/usr/bin/env bash
# E21 — Long-Context / Chunked Prefill Stress Test
# ==================================================
# Goal:   Stress test vLLM with very long sequences (8K–32K tokens).
#         Measure how chunked prefill keeps P99 decode latency stable
#         even as prefill size grows.
#
# IMPORTANT: This is NOT context parallelism.
#   - Context parallelism = splitting long sequences across multiple GPUs
#   - Chunked prefill = splitting large prefill into smaller chunks on ONE GPU
#     to interleave with decode and avoid head-of-line blocking.
#
# Model:  Qwen/Qwen2.5-7B-Instruct (supports up to 128K context natively)
# Run on: RunPod RTX 3090 (single GPU)

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e21_long_context"
mkdir -p "$RESULTS_DIR"

echo "=== E21: Long-Context / Chunked Prefill Stress Test ==="
echo "Model: $MODEL"
echo "Note: chunked prefill ≠ context parallelism. Single GPU only."

run_test() {
  local MAX_LEN=$1
  local CHUNKED=$2
  local CHUNK_SIZE=$3
  local LABEL=$4

  local EXTRA=""
  [[ "$CHUNKED" == "1" ]] && EXTRA="--enable-chunked-prefill --max-num-batched-tokens $CHUNK_SIZE"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len "$MAX_LEN" --dtype float16 \
    --port "$PORT" $EXTRA --disable-log-requests &
  local PID=$!
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

  python phase4_parallelism/e21_long_context_client.py \
    --port "$PORT" --model "$MODEL" \
    --prompt-len "$MAX_LEN" --output-len 128 \
    --num-requests 50 --label "$LABEL" \
    --output-dir "$RESULTS_DIR"

  kill "$PID" 2>/dev/null || true; sleep 5
}

# Short context baseline
run_test 2048  0 0    "2k_no_chunk"
run_test 2048  1 512  "2k_chunked"

# Long context
run_test 8192  0 0    "8k_no_chunk"
run_test 8192  1 512  "8k_chunked_512"
run_test 8192  1 1024 "8k_chunked_1024"

# Very long context
run_test 32768 1 512  "32k_chunked_512"
run_test 32768 1 1024 "32k_chunked_1024"
run_test 32768 1 2048 "32k_chunked_2048"

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: TTFT vs prompt_len, P99 decode latency vs chunk_size."
