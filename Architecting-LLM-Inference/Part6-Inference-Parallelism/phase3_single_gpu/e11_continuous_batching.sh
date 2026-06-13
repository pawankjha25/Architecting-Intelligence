#!/usr/bin/env bash
# E11 — Continuous Batching Deep Dive
# =====================================
# Goal:   Measure real throughput gains from continuous batching under load.
#         vLLM uses continuous batching by default — this shows the effect
#         at increasing request rates and concurrency.
# Run on: RunPod RTX 3090 (single GPU)
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT=8000
RESULTS_DIR="results/e11_continuous_batching"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E11: Continuous Batching ==="
echo "Model: $MODEL"
echo ""

# ── Start vLLM server (continuous batching is default) ────────────────────────
echo "[1/2] Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port "$PORT" \
  --max-num-seqs 256 &
SERVER_PID=$!

echo "  Waiting for server..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  Server ready."
    break
  fi
  sleep 2
done

# ── Rate sweep — shows how continuous batching handles increasing load ─────────
echo ""
echo "[2/2] Running rate sweep..."
python3 phase3_single_gpu/e09_benchmark_client.py \
  --base-url "http://localhost:$PORT" \
  --model "$MODEL" \
  --num-prompts 200 \
  --request-rates 1 5 10 20 50 100 \
  --result-dir "$RESULTS_DIR"

# ── GPU memory snapshot ───────────────────────────────────────────────────────
echo ""
echo "GPU memory snapshot:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu \
           --format=csv

kill "$SERVER_PID" 2>/dev/null || true

# ── Combine all rate results into E11 summary JSON ────────────────────────────
python3 - <<'EOF'
import json, glob, os

results_dir = "results/e11_continuous_batching"
rate_files = sorted(glob.glob(f"{results_dir}/baseline_rps*.json"))
combined_file = f"{results_dir}/E11_baseline.json"

# e09_benchmark_client already saves E09_baseline.json — rename for E11
src = f"{results_dir}/E09_baseline.json"
if os.path.exists(src):
    with open(src) as f:
        data = json.load(f)
    data["experiment_id"] = "E11_continuous_batching"
    data["description"] = "Continuous batching rate sweep — throughput vs latency"
    with open(combined_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {combined_file}")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
