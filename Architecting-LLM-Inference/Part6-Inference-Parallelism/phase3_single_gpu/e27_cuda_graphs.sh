#!/usr/bin/env bash
# E27 — CUDA Graphs: On vs Off
# ==============================
# Goal:   Measure the decode throughput and latency gain from CUDA Graph capture.
#         vLLM enables CUDA Graphs by default for the decode phase.
#         --enforce-eager disables them, forcing eager (step-by-step) execution.
#
# What CUDA Graphs do:
#   - vLLM records the decode forward pass as a CUDA Graph during warmup.
#   - Subsequent decode steps replay the graph via a single CUDA API call.
#   - This eliminates Python/CUDA launch overhead on every decode step.
#   - Effect is strongest for small batch sizes (low concurrency) where
#     kernel launch overhead is a large fraction of per-step time.
#   - At very high batch sizes the gain shrinks (compute dominates overhead).
#
# Run on: RunPod RTX 3090 (single GPU)
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e27_cuda_graphs"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E27: CUDA Graphs On vs Off ==="
echo "Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

run_benchmark() {
  local LABEL=$1
  local EAGER_FLAG="${2:-}"

  echo "--- $LABEL ---"
  local START_TIME=$(date +%s)

  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 4096 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --port "$PORT" \
    $EAGER_FLAG &
  local PID=$!

  echo "  Waiting for server (graph capture may take ~30s)..."
  for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "  Server ready."
      break
    fi
    sleep 2
  done

  local READY_TIME=$(date +%s)
  local STARTUP_S=$((READY_TIME - START_TIME))
  echo "  Startup time: ${STARTUP_S}s"

  # Sweep request rates — low rates show largest CUDA Graph benefit
  python3 phase3_single_gpu/e09_benchmark_client.py \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --num-prompts 200 \
    --request-rates 1 2 4 8 16 32 \
    --result-dir "$RESULTS_DIR/$LABEL"

  # Tag with startup time and cuda_graphs flag
  if [ -f "$RESULTS_DIR/$LABEL/E09_baseline.json" ]; then
    python3 - <<PYEOF
import json
path = "results/e27_cuda_graphs/$LABEL/E09_baseline.json"
with open(path) as f:
    data = json.load(f)
data["cuda_graphs_enabled"] = "$LABEL" == "cuda_graphs_on"
data["startup_time_s"] = $STARTUP_S
data["label"] = "$LABEL"
out = "results/e27_cuda_graphs/${LABEL}.json"
with open(out, "w") as f:
    json.dump(data, f, indent=2)
PYEOF
  fi

  kill "$PID" 2>/dev/null || true
  sleep 5
}

# ── 1. CUDA Graphs enabled (default — no --enforce-eager) ─────────────────────
run_benchmark "cuda_graphs_on" ""

# ── 2. Eager mode (CUDA Graphs disabled) ──────────────────────────────────────
run_benchmark "cuda_graphs_off" "--enforce-eager"

# ── 3. Combine results and print comparison ────────────────────────────────────
python3 - <<'EOF'
import json, os

results_dir = "results/e27_cuda_graphs"

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

on_data  = load(f"{results_dir}/cuda_graphs_on.json")
off_data = load(f"{results_dir}/cuda_graphs_off.json")

if not on_data or not off_data:
    print("Missing result files — skipping comparison")
    exit(0)

on_sweep  = {r["request_rate"]: r for r in on_data.get("rate_sweep", [])}
off_sweep = {r["request_rate"]: r for r in off_data.get("rate_sweep", [])}

comparison = []
for rate in sorted(on_sweep.keys()):
    on_r  = on_sweep[rate]
    off_r = off_sweep.get(rate, {})
    tpot_on  = on_r.get("tpot_ms_mean", 0)
    tpot_off = off_r.get("tpot_ms_mean", 0)
    speedup  = tpot_off / tpot_on if tpot_on > 0 else 0
    comparison.append({
        "request_rate": rate,
        "cuda_graphs_on_tpot_ms": tpot_on,
        "cuda_graphs_off_tpot_ms": tpot_off,
        "cuda_graphs_on_tok_s": on_r.get("output_tokens_per_sec", 0),
        "cuda_graphs_off_tok_s": off_r.get("output_tokens_per_sec", 0),
        "tpot_speedup": round(speedup, 2),
    })

combined = {
    "experiment_id": "E27_cuda_graphs",
    "description": "CUDA Graphs on vs off — TPOT speedup at varying request rates",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "startup_time_with_graphs_s": on_data.get("startup_time_s"),
    "startup_time_without_graphs_s": off_data.get("startup_time_s"),
    "comparison": comparison,
}
out_path = f"{results_dir}/E27_cuda_graphs.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Results saved → {out_path}")

print(f"\n{'Rate':>6} {'On TPOT':>10} {'Off TPOT':>10} {'On Tok/s':>10} {'Off Tok/s':>10} {'Speedup':>9}")
print("-" * 60)
for r in comparison:
    print(f"  {r['request_rate']:>4.0f} {r['cuda_graphs_on_tpot_ms']:>10.1f} "
          f"{r['cuda_graphs_off_tpot_ms']:>10.1f} "
          f"{r['cuda_graphs_on_tok_s']:>10.0f} {r['cuda_graphs_off_tok_s']:>10.0f} "
          f"{r['tpot_speedup']:>8.2f}x")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo ""
echo "Expected pattern:"
echo "  Low RPS  (1–4):   CUDA Graphs give largest speedup (10–30% TPOT reduction)"
echo "  High RPS (16+):   Gain diminishes — compute dominates launch overhead"
