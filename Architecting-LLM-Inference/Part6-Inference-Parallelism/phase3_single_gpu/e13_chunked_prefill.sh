#!/usr/bin/env bash
# E13 — Chunked Prefill
# =======================
# Goal:   Show how chunked prefill reduces P99 decode latency by
#         interleaving large prefill computation with decode steps.
#         Trade-off: TTFT increases slightly, but decode latency stabilizes.
# Run on: RunPod RTX 3090
# Model:  Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
RESULTS_DIR="results/e13_chunked_prefill"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

echo "=== E13: Chunked Prefill ==="
echo "Model: $MODEL"
echo ""

run_config() {
  local LABEL=$1
  local EXTRA_ARGS="${2:-}"

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

  python3 phase3_single_gpu/e09_benchmark_client.py \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --num-prompts 200 \
    --request-rates 10 \
    --result-dir "$RESULTS_DIR/$LABEL"

  # Rename the output JSON with config label
  if [ -f "$RESULTS_DIR/$LABEL/E09_baseline.json" ]; then
    cp "$RESULTS_DIR/$LABEL/E09_baseline.json" "$RESULTS_DIR/${LABEL}.json"
  fi

  kill "$PID" 2>/dev/null || true
  sleep 5
}

# ── No chunked prefill (baseline) ─────────────────────────────────────────────
run_config "no_chunking" ""

# ── Chunked prefill with varying chunk sizes ───────────────────────────────────
for CHUNK in 256 512 1024 2048; do
  echo ""
  echo "  chunk_size=$CHUNK..."
  run_config "chunk_${CHUNK}" "--enable-chunked-prefill --max-num-batched-tokens $CHUNK"
done

# ── Combine all configs into E13 summary JSON ─────────────────────────────────
python3 - <<'EOF'
import json, glob, os

results_dir = "results/e13_chunked_prefill"
configs = ["no_chunking", "chunk_256", "chunk_512", "chunk_1024", "chunk_2048"]

results = []
for cfg in configs:
    path = f"{results_dir}/{cfg}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Extract the rate=10 summary
        rate_data = data.get("rate_sweep", [{}])[0]
        results.append({
            "config": cfg,
            "chunk_size": int(cfg.split("_")[1]) if cfg != "no_chunking" else None,
            "chunked_prefill_enabled": cfg != "no_chunking",
            "request_rate": rate_data.get("request_rate", 10),
            "ttft_ms_mean": rate_data.get("ttft_ms_mean"),
            "ttft_ms_p99": rate_data.get("ttft_ms_p99"),
            "tpot_ms_mean": rate_data.get("tpot_ms_mean"),
            "e2e_ms_p99": rate_data.get("e2e_ms_p99"),
            "output_tokens_per_sec": rate_data.get("output_tokens_per_sec"),
        })

combined = {
    "experiment_id": "E13_chunked_prefill",
    "description": "Chunked prefill — TTFT vs TPOT tradeoff at varying chunk sizes",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "request_rate": 10,
    "configs": results,
}
out_path = f"{results_dir}/E13_chunked_prefill.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Results saved → {out_path}")

# Print summary table
print(f"\n{'Config':<20} {'TTFT_mean':>10} {'TTFT_p99':>10} {'TPOT_mean':>10} {'Tok/s':>8}")
print("-" * 65)
for r in results:
    print(f"  {r['config']:<18} {r['ttft_ms_mean'] or 0:>10.0f} "
          f"{r['ttft_ms_p99'] or 0:>10.0f} {r['tpot_ms_mean'] or 0:>10.0f} "
          f"{r['output_tokens_per_sec'] or 0:>8.0f}")
EOF

echo ""
echo "Done. Results → $RESULTS_DIR/"
echo "Key metrics: TTFT vs chunk size, P99 TPOT vs chunk size."
