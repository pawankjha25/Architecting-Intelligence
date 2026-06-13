#!/usr/bin/env bash
# E18 — Tensor Parallelism TP=2
# ================================
# Goal:   Split model layers across 2 GPUs via tensor parallelism.
#         Use a model that REQUIRES 2 GPUs — forces TP to be necessary,
#         making the experiment more meaningful than TP on a 7B model.
#
# Model choice: Qwen/Qwen2.5-14B-Instruct
#   - 28GB fp16 → requires 2× RTX 3090 (14GB per GPU)
#   - No quantization → clean isolation of TP effects
#   - Apache 2.0, open access
#
# Alt: Qwen/Qwen2.5-32B-Instruct with GPTQ 4-bit if 14B is too easy.
#
# Run on: RunPod 2× RTX 3090

set -euo pipefail

MODEL="Qwen/Qwen2.5-14B-Instruct"
PORT=8000
RESULTS_DIR="results/e18_tensor_parallel"
mkdir -p "$RESULTS_DIR"

echo "=== E18: Tensor Parallelism TP=2 ==="
echo "Model: $MODEL  (requires 2 GPUs)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── TP=1: baseline (14B won't fit on 1 GPU — this will OOM intentionally) ────
echo ""
echo "[1/3] TP=1 (expected OOM — demonstrates why TP is needed)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --max-model-len 4096 --dtype float16 \
  --tensor-parallel-size 1 --port "$PORT" --disable-log-requests 2>&1 | head -20 \
  || echo "  OOM as expected — model does not fit on 1 GPU."
sleep 3

# ── TP=2: model split across both GPUs ───────────────────────────────────────
echo ""
echo "[2/3] TP=2 (model sharded across GPU 0 and GPU 1)..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --port "$PORT" \
  --disable-log-requests &
PID=$!

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
echo "  Server ready."

# GPU memory snapshot (should show ~14GB per device)
echo "  GPU memory after loading:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo ""
echo "[3/3] Benchmarking TP=2..."
for RPS in 1 5 10 20; do
  python -m vllm.benchmarks.benchmark_serving \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name sharegpt --num-prompts 200 --request-rate "$RPS" \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "tp2_rps${RPS}.json"
done

kill "$PID" 2>/dev/null || true

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: TTFT, TPOT, throughput vs single-GPU baseline (E09)."
echo "Note: TP reduces per-device memory but adds all-reduce communication overhead."
