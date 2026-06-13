#!/usr/bin/env bash
# E16 — Multi-LoRA Serving (S-LoRA style)
# =========================================
# Goal:   Serve N LoRA adapters from one base model instance.
#         Show latency and memory overhead vs single adapter.
# Run on: RunPod RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct (base)
# LoRAs:  Community adapters from HuggingFace (no gating)

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e16_multi_lora"
mkdir -p "$RESULTS_DIR"

# Public LoRA adapters for Qwen2.5-7B (verify availability before running)
# Replace with any compatible LoRA adapters you have access to.
LORA_1="predibase/qwen2.5-7b-instruct-codeAlpaca"
LORA_2="predibase/qwen2.5-7b-instruct-sqlCoder"

echo "=== E16: Multi-LoRA Serving ==="
echo "Base model: $MODEL"

# ── Baseline: base model only ─────────────────────────────────────────────────
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --port "$PORT" --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 100 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "base_only.json"
kill "$PID" 2>/dev/null || true; sleep 5

# ── Multi-LoRA: 2 adapters ────────────────────────────────────────────────────
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "adapter1=$LORA_1" "adapter2=$LORA_2" \
  --max-loras 2 \
  --max-lora-rank 64 \
  --disable-log-requests &
PID=$!
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done

python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:$PORT" --model "adapter1" \
  --dataset-name sharegpt --num-prompts 100 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "multi_lora_2adapters.json"
kill "$PID" 2>/dev/null || true

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: latency overhead per adapter, GPU memory delta, throughput."
