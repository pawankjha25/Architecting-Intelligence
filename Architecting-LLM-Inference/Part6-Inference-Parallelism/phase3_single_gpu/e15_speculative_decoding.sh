#!/usr/bin/env bash
# E15 — Speculative Decoding
# ============================
# Goal:   Measure token acceptance rate and latency speedup.
#         Draft model proposes K tokens; target verifies in parallel.
# Run on: RunPod RTX 3090
# Models: Draft: Qwen/Qwen2.5-1.5B-Instruct  Target: Qwen/Qwen2.5-7B-Instruct
#         Same family → higher acceptance rate vs cross-family pairing.

set -euo pipefail

TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
RESULTS_DIR="results/e15_speculative_decoding"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E15: Speculative Decoding ==="
echo "Target: $TARGET_MODEL  |  Draft: $DRAFT_MODEL"
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

# ── Baseline: target model alone ─────────────────────────────────────────────
echo "[1/2] Baseline (target only)..."
python3 -m vllm.entrypoints.openai.api_server \
  --model "$TARGET_MODEL" \
  --max-model-len 4096 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port "$PORT" &
PID=$!
wait_for_server

python3 phase3_single_gpu/e09_benchmark_client.py \
  --base-url "http://localhost:$PORT" \
  --model "$TARGET_MODEL" \
  --num-prompts 200 \
  --request-rates 5 \
  --result-dir "$RESULTS_DIR/baseline"

cp "$RESULTS_DIR/baseline/E09_baseline.json" "$RESULTS_DIR/baseline.json" 2>/dev/null || true
kill "$PID" 2>/dev/null || true; sleep 5

# ── Speculative decoding: vary num_speculative_tokens ─────────────────────────
echo ""
echo "[2/2] Speculative decoding (K token sweep)..."
for K in 3 5 7; do
  echo ""
  echo "  num_speculative_tokens=$K..."
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --speculative-model "$DRAFT_MODEL" \
    --num-speculative-tokens "$K" \
    --max-model-len 4096 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port "$PORT" &
  PID=$!
  wait_for_server || { kill "$PID" 2>/dev/null; continue; }

  python3 phase3_single_gpu/e09_benchmark_client.py \
    --base-url "http://localhost:$PORT" \
    --model "$TARGET_MODEL" \
    --num-prompts 200 \
    --request-rates 5 \
    --result-dir "$RESULTS_DIR/spec_k${K}"

  if [ -f "$RESULTS_DIR/spec_k${K}/E09_baseline.json" ]; then
    python3 - <<PYEOF
import json
path = "results/e15_speculative_decoding/spec_k${K}/E09_baseline.json"
with open(path) as f:
    data = json.load(f)
data["num_speculative_tokens"] = $K
data["draft_model"] = "$DRAFT_MODEL"
out = "results/e15_speculative_decoding/spec_k${K}.json"
with open(out, "w") as f:
    json.dump(data, f, indent=2)
PYEOF
  fi

  kill "$PID" 2>/dev/null || true; sleep 5
done

# ── Combine all results into E15 summary JSON ─────────────────────────────────
python3 - <<'EOF'
import json, os

results_dir = "results/e15_speculative_decoding"

def load_rate(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    sweep = data.get("rate_sweep", [{}])
    return sweep[0] if sweep else {}

baseline = load_rate(f"{results_dir}/baseline.json")
spec_results = []
for k in [3, 5, 7]:
    r = load_rate(f"{results_dir}/spec_k{k}.json")
    if r:
        base_tpot = baseline.get("tpot_ms_mean", 1)
        spec_tpot = r.get("tpot_ms_mean", 1)
        speedup = base_tpot / spec_tpot if spec_tpot else 0
        spec_results.append({
            "num_speculative_tokens": k,
            "tpot_ms_mean": r.get("tpot_ms_mean"),
            "ttft_ms_mean": r.get("ttft_ms_mean"),
            "output_tokens_per_sec": r.get("output_tokens_per_sec"),
            "tpot_speedup_vs_baseline": round(speedup, 2),
        })

combined = {
    "experiment_id": "E15_speculative_decoding",
    "description": "Speculative decoding — TPOT speedup vs num_speculative_tokens K",
    "target_model": "Qwen/Qwen2.5-7B-Instruct",
    "draft_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "request_rate": 5,
    "baseline": {
        "tpot_ms_mean": baseline.get("tpot_ms_mean"),
        "ttft_ms_mean": baseline.get("ttft_ms_mean"),
        "output_tokens_per_sec": baseline.get("output_tokens_per_sec"),
    },
    "speculative_results": spec_results,
}
out_path = f"{results_dir}/E15_speculative_decoding.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Results saved → {out_path}")

print(f"\n{'Config':<20} {'TPOT_mean':>10} {'Tok/s':>8} {'Speedup':>9}")
print("-" * 52)
print(f"  {'baseline':<18} {baseline.get('tpot_ms_mean', 0):>10.1f} "
      f"{baseline.get('output_tokens_per_sec', 0):>8.0f} {'1.00x':>9}")
for r in spec_results:
    print(f"  {'spec_k=' + str(r['num_speculative_tokens']):<18} "
          f"{r['tpot_ms_mean'] or 0:>10.1f} "
          f"{r['output_tokens_per_sec'] or 0:>8.0f} "
          f"{r['tpot_speedup_vs_baseline']:>8.2f}x")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: TPOT speedup vs K, acceptance rate sensitivity."
