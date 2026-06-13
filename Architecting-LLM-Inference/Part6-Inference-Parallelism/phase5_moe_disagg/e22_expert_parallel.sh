#!/usr/bin/env bash
# E22 — Expert Parallelism (MoE)
# ================================
# Goal:   Show how expert parallelism distributes MoE experts across GPUs.
#         Measure GPU load balance, expert utilization, throughput.
# Run on: RunPod 2× RTX 3090
# Model:  mistralai/Mixtral-8x7B-Instruct-v0.1 GPTQ 4-bit (~24GB total)
#         With TP=2: ~12GB per GPU — fits on 2× RTX 3090

set -euo pipefail

MODEL="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"   # 4-bit GPTQ, open access
PORT=8000
RESULTS_DIR="results/e22_expert_parallel"
mkdir -p "$RESULTS_DIR"

echo "=== E22: Expert Parallelism (Mixtral 8x7B GPTQ) ==="
echo "Model: $MODEL"
echo "Note: vLLM uses TP for expert distribution in MoE models."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── TP=2 (expert parallelism via tensor parallel) ─────────────────────────────
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization gptq \
  --max-model-len 4096 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --port "$PORT" \
  --disable-log-requests &
PID=$!

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
echo "Server ready."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for RPS in 1 5 10 20; do
  python -m vllm.benchmarks.benchmark_serving \
    --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name sharegpt --num-prompts 200 --request-rate "$RPS" \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "moe_rps${RPS}.json"
done

kill "$PID" 2>/dev/null || true

echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: GPU utilization per device, throughput vs dense Qwen2.5-14B (E18)."
