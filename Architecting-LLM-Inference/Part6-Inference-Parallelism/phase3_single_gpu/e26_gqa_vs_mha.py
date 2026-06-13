"""
E26 — GQA vs MHA: KV Cache Memory and Throughput
===================================================
Goal:   Show concretely how Grouped Query Attention (GQA) reduces KV cache
        memory vs Multi-Head Attention (MHA), and what that means for
        throughput and max concurrent requests.

Background:
  MHA: num_kv_heads == num_q_heads  (e.g. GPT-2: 12 Q heads, 12 KV heads)
  GQA: num_kv_heads <  num_q_heads  (e.g. Qwen2.5-7B: 28 Q heads, 8 KV heads)
  MQA: num_kv_heads == 1            (extreme case — one KV head shared by all)

  KV cache bytes per token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
  GQA reduces this by (num_q_heads / num_kv_heads) × compared to MHA.

Phase A — MacBook: Pure simulation and math
Phase B — RunPod:  Real vLLM benchmark comparing two models

Models for Phase B:
  MHA (approximate): gpt2-xl  (25 Q heads, 25 KV heads, 48 layers)
  GQA:               Qwen/Qwen2.5-7B-Instruct (28 Q heads, 8 KV heads, 28 layers)

Run Phase A on MacBook. Run Phase B on RunPod RTX 3090.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path("results/e26_gqa_vs_mha")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model attention configs ────────────────────────────────────────────────────

@dataclass
class AttentionConfig:
    name: str
    attention_type: str        # "MHA", "GQA", "MQA"
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    dtype_bytes: int = 2       # fp16
    total_params_b: float = 0.0

    @property
    def kv_heads_ratio(self) -> float:
        return self.num_kv_heads / self.num_q_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache consumed per token (both K and V, all layers)."""
        return 2 * self.num_kv_heads * self.head_dim * self.num_layers * self.dtype_bytes

    @property
    def kv_reduction_vs_mha(self) -> float:
        """How much smaller KV cache is vs equivalent MHA."""
        return self.num_q_heads / self.num_kv_heads

    def max_tokens_in_vram(self, vram_gb: float, model_weights_gb: float) -> int:
        """How many tokens fit in KV cache after model weights."""
        available = max(0.0, vram_gb - model_weights_gb) * 1e9
        return int(available / self.kv_bytes_per_token)


# Representative configs
CONFIGS = [
    AttentionConfig("GPT-2 XL",           "MHA", num_q_heads=25, num_kv_heads=25, head_dim=64,  num_layers=48, total_params_b=1.5),
    AttentionConfig("LLaMA-1 7B",         "MHA", num_q_heads=32, num_kv_heads=32, head_dim=128, num_layers=32, total_params_b=7.0),
    AttentionConfig("Mistral 7B",         "GQA", num_q_heads=32, num_kv_heads=8,  head_dim=128, num_layers=32, total_params_b=7.0),
    AttentionConfig("Qwen2.5-7B",         "GQA", num_q_heads=28, num_kv_heads=8,  head_dim=128, num_layers=28, total_params_b=7.0),
    AttentionConfig("Qwen2.5-14B",        "GQA", num_q_heads=40, num_kv_heads=8,  head_dim=128, num_layers=40, total_params_b=14.0),
    AttentionConfig("Qwen2.5-72B",        "GQA", num_q_heads=64, num_kv_heads=8,  head_dim=128, num_layers=80, total_params_b=72.0),
    AttentionConfig("Hypothetical MQA 7B","MQA", num_q_heads=32, num_kv_heads=1,  head_dim=128, num_layers=32, total_params_b=7.0),
]

VRAM_GB = 24.0   # RTX 3090


# ── Phase A: Simulation ────────────────────────────────────────────────────────

def phase_a_analysis():
    print("Phase A — KV Cache Memory Analysis (no GPU needed)\n")
    print(f"Hardware: {VRAM_GB}GB VRAM (RTX 3090)\n")

    print(f"{'Model':<22} {'Type':>5} {'KV heads':>9} {'KV B/tok':>10} "
          f"{'Reduction':>10} {'Max tokens':>12} {'Max seqs@512':>14}")
    print("─" * 90)

    results = []
    for cfg in CONFIGS:
        model_gb = cfg.total_params_b * cfg.dtype_bytes  # rough: 2B per param for fp16
        max_tokens = cfg.max_tokens_in_vram(VRAM_GB, model_gb)
        max_seqs_512 = max_tokens // 512   # how many 512-token sequences fit

        print(f"  {cfg.name:<20} {cfg.attention_type:>5} "
              f"  {cfg.num_kv_heads}/{cfg.num_q_heads:>2}"
              f"  {cfg.kv_bytes_per_token:>8,}B "
              f"  {cfg.kv_reduction_vs_mha:>8.1f}x "
              f"  {max_tokens:>10,} "
              f"  {max_seqs_512:>12}")

        results.append({
            "model": cfg.name,
            "type": cfg.attention_type,
            "num_q_heads": cfg.num_q_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "kv_bytes_per_token": cfg.kv_bytes_per_token,
            "kv_reduction_vs_mha": cfg.kv_reduction_vs_mha,
            "max_tokens_24gb": max_tokens,
            "max_seqs_512tok": max_seqs_512,
        })

    # Save
    with open(RESULTS_DIR / "simulation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Key comparison: LLaMA-1 7B (MHA) vs Qwen2.5-7B (GQA)
    mha = next(r for r in results if r["model"] == "LLaMA-1 7B")
    gqa = next(r for r in results if r["model"] == "Qwen2.5-7B")

    print(f"\n── MHA vs GQA head-to-head (7B scale, {VRAM_GB}GB VRAM) ─────────────")
    print(f"  KV bytes/token: MHA={mha['kv_bytes_per_token']:,}B  GQA={gqa['kv_bytes_per_token']:,}B  "
          f"→ {mha['kv_bytes_per_token']/gqa['kv_bytes_per_token']:.1f}x less memory per token")
    print(f"  Max seqs@512:   MHA={mha['max_seqs_512tok']}  GQA={gqa['max_seqs_512tok']}  "
          f"→ {gqa['max_seqs_512tok']/max(mha['max_seqs_512tok'],1):.1f}x more concurrent requests")


def phase_a_throughput_model():
    """
    Model how GQA affects throughput.
    Memory-bound decode: throughput ∝ 1 / kv_bytes_per_token (more concurrent → more throughput).
    """
    print("\n── Throughput Model (memory-bound decode regime) ─────────────────────")
    print("  Assumption: decode is memory-bandwidth bound.")
    print("  Throughput ∝ concurrent sequences × tokens/step")
    print(f"  GPU HBM bandwidth: 936 GB/s (RTX 3090)\n")

    HBM_BW_GBS = 936.0
    BASE_TPOT_MS = 15.0   # baseline single-sequence TPOT

    print(f"  {'Model':<22} {'Concurrency':>12} {'Rel. throughput':>16}")
    mha_conc = None
    for cfg in CONFIGS:
        model_gb = cfg.total_params_b * cfg.dtype_bytes
        max_tokens = cfg.max_tokens_in_vram(VRAM_GB, model_gb)
        concurrency = max_tokens // 256   # 256-token avg sequence

        if mha_conc is None:
            mha_conc = max(concurrency, 1)

        rel = concurrency / mha_conc
        print(f"  {cfg.name:<22}   {concurrency:>10}  {rel:>14.1f}x")


# ── Phase B: Real vLLM benchmark ───────────────────────────────────────────────

def phase_b_vllm(port: int = 8000):
    """
    Run on RunPod. Compares KV cache utilization and throughput between
    a model with many KV heads vs few KV heads.

    Since we're standardizing on Qwen2.5, we compare:
      - Qwen2.5-7B-Instruct:  8 KV heads (GQA)
      - gpt2-xl:             25 KV heads (MHA, much smaller model but same principle)

    The key measurement: how many concurrent requests each can sustain
    before KV cache pressure causes degradation.
    """
    import subprocess
    import requests as req_lib
    import asyncio
    import aiohttp

    MODELS = [
        ("Qwen/Qwen2.5-7B-Instruct", "GQA (8 KV heads)"),
        ("gpt2-xl",                   "MHA (25 KV heads)"),
    ]

    all_results = {}

    for model_id, label in MODELS:
        print(f"\n  Starting vLLM with {label} ({model_id})...")
        proc = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--max-model-len", "4096",
            "--dtype", "float16",
            "--port", str(port),
            "--disable-log-requests",
        ])

        # Wait for server
        for _ in range(60):
            try:
                if req_lib.get(f"http://localhost:{port}/health", timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(2)

        # Query KV cache stats via vLLM metrics endpoint
        try:
            metrics = req_lib.get(f"http://localhost:{port}/metrics").text
            # Parse gpu_cache_usage_perc from prometheus metrics
            for line in metrics.splitlines():
                if "gpu_cache_usage_perc" in line and not line.startswith("#"):
                    print(f"    GPU KV cache usage: {line.split()[-1]}")
        except Exception:
            pass

        # Benchmark at different concurrency levels
        model_results = {}
        for concurrency in [1, 8, 32, 64]:
            result = subprocess.run([
                "python", "-m", "vllm.benchmarks.benchmark_serving",
                "--backend", "openai",
                "--base-url", f"http://localhost:{port}",
                "--model", model_id,
                "--dataset-name", "sharegpt",
                "--num-prompts", "100",
                "--request-rate", str(concurrency),
                "--percentile-metrics", "ttft,tpot,e2e_latency",
            ], capture_output=True, text=True)

            model_results[f"rps_{concurrency}"] = result.stdout
            print(f"    concurrency={concurrency}: done")

        all_results[label] = model_results
        proc.terminate()
        proc.wait()
        time.sleep(5)

    with open(RESULTS_DIR / "vllm_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results → {RESULTS_DIR}/vllm_benchmark.json")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("E26 — GQA vs MHA: KV Cache Memory and Throughput\n")

    phase_a_analysis()
    phase_a_throughput_model()

    print("\n" + "─" * 70)
    print("Phase B requires RunPod RTX 3090. Run with --runpod flag:")
    print("  python e26_gqa_vs_mha.py --runpod")
    print("─" * 70)

    import sys
    if "--runpod" in sys.argv:
        phase_b_vllm()

    print("\n[Article insight]")
    print("  GQA reduces KV heads from num_q_heads → num_kv_heads (e.g. 32 → 8).")
    print("  KV cache memory drops by 4x at 7B scale compared to MHA.")
    print("  4x less KV memory → 4x more concurrent sequences in same VRAM.")
    print("  4x more concurrency → proportionally higher throughput in decode.")
    print("  Quality loss from GQA vs MHA is minimal — standard for all modern LLMs.")
    print("  MQA (1 KV head) pushes this further but quality degrades more noticeably.")


if __name__ == "__main__":
    main()
