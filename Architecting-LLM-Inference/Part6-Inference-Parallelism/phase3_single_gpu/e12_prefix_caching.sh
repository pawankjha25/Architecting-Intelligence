#!/usr/bin/env bash
# E12 — Prefix Caching
# ======================
# Goal:   Measure TTFT reduction from shared prefix reuse.
#         Compares same workload with/without --enable-prefix-caching.
# Run on: RunPod RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e12_prefix_caching"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E12: Prefix Caching ==="
echo "Model: $MODEL"
echo ""

run_benchmark() {
  local PREFIX_CACHING=$1
  local LABEL=$2
  local EXTRA_ARGS=""
  [[ "$PREFIX_CACHING" == "1" ]] && EXTRA_ARGS="--enable-prefix-caching"

  echo "--- $LABEL ---"
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 4096 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port "$PORT" \
    $EXTRA_ARGS &
  local PID=$!

  echo "  Waiting for server..."
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "  Server ready."
      break
    fi
    sleep 2
  done

  python3 phase3_single_gpu/e12_prefix_caching_client.py \
    --port "$PORT" \
    --model "$MODEL" \
    --label "$LABEL" \
    --output-dir "$RESULTS_DIR"

  kill "$PID" 2>/dev/null || true
  sleep 5
}

run_benchmark 0 "no_prefix_cache"
run_benchmark 1 "with_prefix_cache"

# ── Combine both results into single E12 summary JSON ─────────────────────────
python3 - <<'EOF'
import json, os

results_dir = "results/e12_prefix_caching"
labels = ["no_prefix_cache", "with_prefix_cache"]
results = {}

for label in labels:
    path = f"{results_dir}/{label}.json"
    if os.path.exists(path):
        with open(path) as f:
            results[label] = json.load(f)

if len(results) == 2:
    no_cache  = results["no_prefix_cache"]["latency_mean_ms"]
    with_cache = results["with_prefix_cache"]["latency_mean_ms"]
    ttft_reduction_pct = (no_cache - with_cache) / no_cache * 100

    combined = {
        "experiment_id": "E12_prefix_caching",
        "description": "Prefix caching TTFT reduction — shared system prompt workload",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "system_prompt_tokens": results["no_prefix_cache"].get("system_prompt_tokens", 512),
        "num_requests": results["no_prefix_cache"].get("num_requests", 100),
        "no_prefix_cache": results["no_prefix_cache"],
        "with_prefix_cache": results["with_prefix_cache"],
        "ttft_reduction_pct": round(ttft_reduction_pct, 1),
    }
    with open(f"{results_dir}/E12_prefix_caching.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"TTFT reduction: {ttft_reduction_pct:.1f}%")
    print(f"Results saved → {results_dir}/E12_prefix_caching.json")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo "Key metric: TTFT reduction % for requests with shared system prompt."
