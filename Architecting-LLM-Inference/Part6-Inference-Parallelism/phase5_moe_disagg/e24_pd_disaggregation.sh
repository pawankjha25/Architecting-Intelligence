#!/usr/bin/env bash
# E24 — Prefill / Decode Disaggregation (Simulated)
# ====================================================
# Goal:   Simulate P/D split on 2 GPUs.
#         GPU 0 = prefill-only instance (handles first forward pass)
#         GPU 1 = decode-only instance (handles autoregressive generation)
#         Show: prefill GPU stays compute-bound, decode GPU stays memory-bound.
#
# Implementation: Two vLLM instances with different max_num_seqs settings
#   to approximate the P/D role split. True P/D disaggregation with KV transfer
#   requires vLLM's --kv-transfer-config (experimental, v0.6+).
#
# Model:  Qwen/Qwen2.5-7B-Instruct
# Run on: RunPod 2× RTX 3090

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/e24_pd_disaggregation"
mkdir -p "$RESULTS_DIR"

echo "=== E24: Prefill/Decode Disaggregation ==="
echo "Model: $MODEL"
echo ""
echo "Architecture:"
echo "  GPU 0 (port 8000): Prefill-optimized  --max-num-seqs 1  --max-num-batched-tokens 8192"
echo "  GPU 1 (port 8001): Decode-optimized   --max-num-seqs 256 --max-num-batched-tokens 256"
echo ""

# ── Prefill instance (GPU 0) ──────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 8192 --dtype float16 \
  --port 8000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --disable-log-requests &
PREFILL_PID=$!

# ── Decode instance (GPU 1) ───────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 8192 --dtype float16 \
  --port 8001 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 256 \
  --disable-log-requests &
DECODE_PID=$!

until curl -sf "http://localhost:8000/health" > /dev/null 2>&1; do sleep 2; done
until curl -sf "http://localhost:8001/health" > /dev/null 2>&1; do sleep 2; done
echo "Both instances ready."

# ── Baseline: single unified server ──────────────────────────────────────────
echo "[1/3] Measuring single unified server (GPU 0)..."
python -m vllm.benchmarks.benchmark_serving \
  --backend openai --base-url "http://localhost:8000" --model "$MODEL" \
  --dataset-name sharegpt --num-prompts 200 --request-rate 10 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename "unified.json"

# ── Prefill-optimized: long prompts ──────────────────────────────────────────
echo "[2/3] Measuring prefill instance with long prompts..."
python phase5_moe_disagg/e24_pd_client.py \
  --prefill-port 8000 --decode-port 8001 \
  --model "$MODEL" --output-dir "$RESULTS_DIR"

# ── GPU utilization snapshot ──────────────────────────────────────────────────
echo "[3/3] GPU utilization:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

kill "$PREFILL_PID" "$DECODE_PID" 2>/dev/null || true

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo ""
echo "Article insight:"
echo "  Prefill is compute-bound (flops/byte ratio high): benefits from fewer, larger batches."
echo "  Decode is memory-bound (flops/byte ratio low): benefits from many concurrent sequences."
echo "  Disaggregation lets each role be tuned independently."
echo "  In production (DistServe, Mooncake), KV cache is transferred over high-bandwidth fabric."
