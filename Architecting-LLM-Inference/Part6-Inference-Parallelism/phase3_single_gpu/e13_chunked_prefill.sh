#!/usr/bin/env bash
# E13 — Chunked Prefill
# =======================
# Goal:   Show how chunked prefill reduces P99 decode latency by
#         interleaving large prefill computation with decode steps.
#         Trade-off: TTFT increases slightly, but decode latency stabilizes.
# Run on: RunPod RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e13_chunked_prefill"
mkdir -p "$RESULTS_DIR"

echo "=== E13: Chunked Prefill ==="

run_with_chunk_size() {
  local CHUNK_SIZE=$1
  local EXTRA_ARGS="--enable-chunked-prefill"
  [[ "$CHUNK_SIZE" -gt 0 ]] && EXTRA_ARGS+=" --max-num-batched-tokens $CHUNK_SIZE"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --max-model-len 8192 --dtype float16 --port "$PORT" \
    $EXTRA_ARGS --disable-log-requests &
  local PID=$!
  until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

  python -m vllm.benchmarks.benchmark_serving \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name sharegpt --num-prompts 200 --request-rate 10 \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "chunk_${CHUNK_SIZE}.json"

  kill "$PID" 2>/dev/null || true; sleep 3
}

# No chunked prefill (baseline)
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 8192 --dtype float16 \
  --port "$PORT" --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 200 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "no_chunking.json"
kill "$PID" 2>/dev/null || true; sleep 3

# Chunked prefill with varying chunk sizes
for CHUNK in 256 512 1024 2048; do
  echo "  chunk_size=$CHUNK..."
  run_with_chunk_size "$CHUNK"
done

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: TTFT vs chunk size, P99 TPOT vs chunk size."
echo "Note: chunked prefill ≠ context parallelism. It splits prefill computation"
echo "      into chunks and batches them with decode — all on one GPU."
