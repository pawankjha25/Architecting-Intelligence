#!/usr/bin/env bash
# E14 — KV Cache CPU Offloading
# ================================
# Goal:   Show how --cpu-offload-gb extends effective KV capacity
#         at the cost of throughput (PCIe bandwidth bottleneck).
# Run on: RunPod RTX 3090 (24GB GPU + CPU RAM)
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e14_kv_offloading"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E14: KV Cache CPU Offloading ==="
echo "Model: $MODEL"
echo ""

run_with_offload() {
  local CPU_OFFLOAD_GB=$1
  local LABEL="offload_${CPU_OFFLOAD_GB}gb"
  local EXTRA_ARGS=""
  [[ "$CPU_OFFLOAD_GB" -gt 0 ]] && EXTRA_ARGS="--cpu-offload-gb $CPU_OFFLOAD_GB"

  echo "--- cpu_offload_gb=$CPU_OFFLOAD_GB ---"
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
  for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "  Server ready."
      break
    fi
    sleep 2
  done

  # Record GPU memory before benchmark
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits 2>/dev/null || echo "0, 0")
  echo "  GPU memory: $GPU_MEM MiB (used, free)"

  python3 phase3_single_gpu/e09_benchmark_client.py \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --num-prompts 100 \
    --request-rates 5 \
    --result-dir "$RESULTS_DIR/$LABEL"

  # Tag result with config info
  if [ -f "$RESULTS_DIR/$LABEL/E09_baseline.json" ]; then
    python3 - <<PYEOF
import json
path = "results/e14_kv_offloading/$LABEL/E09_baseline.json"
with open(path) as f:
    data = json.load(f)
data["cpu_offload_gb"] = $CPU_OFFLOAD_GB
data["gpu_memory_csv"] = "$GPU_MEM"
out = "results/e14_kv_offloading/${LABEL}.json"
with open(out, "w") as f:
    json.dump(data, f, indent=2)
PYEOF
  fi

  kill "$PID" 2>/dev/null || true
  sleep 5
}

for OFFLOAD_GB in 0 4 8 16; do
  echo ""
  run_with_offload "$OFFLOAD_GB"
done

# ── Combine all offload configs into E14 summary JSON ─────────────────────────
python3 - <<'EOF'
import json, glob, os

results_dir = "results/e14_kv_offloading"
offload_gbs = [0, 4, 8, 16]

results = []
for gb in offload_gbs:
    path = f"{results_dir}/offload_{gb}gb.json"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    rate_data = data.get("rate_sweep", [{}])[0]
    results.append({
        "cpu_offload_gb": gb,
        "request_rate": rate_data.get("request_rate", 5),
        "ttft_ms_mean": rate_data.get("ttft_ms_mean"),
        "ttft_ms_p99": rate_data.get("ttft_ms_p99"),
        "tpot_ms_mean": rate_data.get("tpot_ms_mean"),
        "e2e_ms_p99": rate_data.get("e2e_ms_p99"),
        "output_tokens_per_sec": rate_data.get("output_tokens_per_sec"),
        "gpu_memory_csv": data.get("gpu_memory_csv", ""),
    })

combined = {
    "experiment_id": "E14_kv_offloading",
    "description": "KV cache CPU offloading — throughput vs offload GB (PCIe bottleneck)",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "hardware": "RTX 3090 24GB + CPU RAM",
    "request_rate": 5,
    "configs": results,
}
out_path = f"{results_dir}/E14_kv_offloading.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Results saved → {out_path}")

print(f"\n{'Offload GB':>12} {'TTFT_mean':>10} {'TPOT_mean':>10} {'Tok/s':>8}")
print("-" * 45)
for r in results:
    print(f"  {r['cpu_offload_gb']:>10}GB {r['ttft_ms_mean'] or 0:>10.0f} "
          f"{r['tpot_ms_mean'] or 0:>10.0f} {r['output_tokens_per_sec'] or 0:>8.0f}")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: throughput penalty and TTFT increase per GB offloaded."
