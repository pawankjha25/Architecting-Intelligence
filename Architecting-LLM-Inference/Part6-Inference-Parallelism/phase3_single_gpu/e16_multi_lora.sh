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

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

# Public LoRA adapters for Qwen2.5-7B
LORA_1="predibase/qwen2.5-7b-instruct-codeAlpaca"
LORA_2="predibase/qwen2.5-7b-instruct-sqlCoder"

echo "=== E16: Multi-LoRA Serving ==="
echo "Base model: $MODEL"
echo ""

wait_for_server() {
  echo "  Waiting for server..."
  for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "  Server ready."
      return 0
    fi
    sleep 2
  done
  echo "  ERROR: server did not start"
  return 1
}

# ── Baseline: base model only ─────────────────────────────────────────────────
echo "[1/2] Baseline (base model only)..."
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port "$PORT" &
PID=$!
wait_for_server

GPU_MEM_BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")

python3 phase3_single_gpu/e09_benchmark_client.py \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --num-prompts 100 \
  --request-rates 10 \
  --result-dir "$RESULTS_DIR/base_only"

cp "$RESULTS_DIR/base_only/E09_baseline.json" "$RESULTS_DIR/base_only.json" 2>/dev/null || true
kill "$PID" 2>/dev/null || true; sleep 5

# ── Multi-LoRA: 2 adapters ────────────────────────────────────────────────────
echo ""
echo "[2/2] Multi-LoRA (2 adapters)..."
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "adapter1=$LORA_1" "adapter2=$LORA_2" \
  --max-loras 2 \
  --max-lora-rank 64 &
PID=$!
wait_for_server || { kill "$PID" 2>/dev/null || true; echo "  Skipping LoRA test — server failed to start"; PID=""; }

if [ -n "${PID:-}" ]; then
  GPU_MEM_LORA=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")

  python3 phase3_single_gpu/e09_benchmark_client.py \
    --base-url "http://localhost:$PORT" \
    --model "adapter1" \
    --num-prompts 100 \
    --request-rates 10 \
    --result-dir "$RESULTS_DIR/multi_lora_2adapters"

  cp "$RESULTS_DIR/multi_lora_2adapters/E09_baseline.json" \
     "$RESULTS_DIR/multi_lora_2adapters.json" 2>/dev/null || true
  kill "$PID" 2>/dev/null || true
else
  GPU_MEM_LORA="0"
fi

# ── Combine results into E16 summary JSON ────────────────────────────────────
python3 - <<PYEOF
import json, os

results_dir = "results/e16_multi_lora"

def load_rate(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    sweep = data.get("rate_sweep", [{}])
    return sweep[0] if sweep else {}

base = load_rate(f"{results_dir}/base_only.json")
lora = load_rate(f"{results_dir}/multi_lora_2adapters.json")

combined = {
    "experiment_id": "E16_multi_lora",
    "description": "Multi-LoRA serving — latency and memory overhead vs base model",
    "base_model": "$MODEL",
    "lora_adapters": ["$LORA_1", "$LORA_2"],
    "request_rate": 10,
    "gpu_memory_base_mib": $GPU_MEM_BASE,
    "gpu_memory_lora_mib": $GPU_MEM_LORA,
    "gpu_memory_overhead_mib": $GPU_MEM_LORA - $GPU_MEM_BASE,
    "base_only": {
        "tpot_ms_mean": base.get("tpot_ms_mean"),
        "ttft_ms_mean": base.get("ttft_ms_mean"),
        "output_tokens_per_sec": base.get("output_tokens_per_sec"),
    },
    "multi_lora_2adapters": {
        "tpot_ms_mean": lora.get("tpot_ms_mean"),
        "ttft_ms_mean": lora.get("ttft_ms_mean"),
        "output_tokens_per_sec": lora.get("output_tokens_per_sec"),
    } if lora else None,
}
out_path = f"{results_dir}/E16_multi_lora.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Results saved → {out_path}")
print(f"GPU memory overhead for 2 LoRA adapters: {combined['gpu_memory_overhead_mib']} MiB")
PYEOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: latency overhead per adapter, GPU memory delta, throughput."
