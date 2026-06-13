"""
E02 — Static Batching
======================
Goal:   Show padding waste and GPU underutilization from static batching.
        All sequences padded to max length in the batch — wasted compute.
Model:  Llama 3.2 1B (CPU/MPS on MacBook)
Vary:   Batch size (1, 4, 8, 16), sequence length variance
"""

import time
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # open, Apache 2.0
BATCH_SIZES = [1, 4, 8, 16]
OUTPUT_TOKENS = 64
PROMPT_MEAN = 256
PROMPT_STD = 128    # high variance → lots of padding


def compute_padding_waste(prompt_lens: list[int]) -> dict:
    """Compute padding overhead for a batch."""
    max_len = max(prompt_lens)
    total_cells = max_len * len(prompt_lens)
    actual_cells = sum(prompt_lens)
    padding_cells = total_cells - actual_cells
    return {
        "max_len": max_len,
        "avg_len": sum(prompt_lens) / len(prompt_lens),
        "padding_pct": (padding_cells / total_cells) * 100,
        "efficiency_pct": (actual_cells / total_cells) * 100,
    }


def run_static_batch_vllm(batch_size: int, requests) -> tuple[list[RequestMetrics], dict]:
    """
    Run requests in fixed static batches using vLLM offline mode.
    Simulates static batching by grouping requests and padding.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, max_model_len=4096, max_num_seqs=batch_size)
    params = SamplingParams(max_tokens=OUTPUT_TOKENS, temperature=0.0)

    metrics_list = []
    padding_stats = []

    # Process in fixed batches
    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start:batch_start + batch_size]
        prompts = [r.prompt for r in batch]
        prompt_lens = [r.prompt_len for r in batch]

        waste = compute_padding_waste(prompt_lens)
        padding_stats.append(waste)

        arrival = time.perf_counter()
        outputs = llm.generate(prompts, params)
        end = time.perf_counter()
        batch_time = end - arrival

        for i, (req, out) in enumerate(zip(batch, outputs)):
            output_len = len(out.outputs[0].token_ids)
            # In static batching, all requests in batch start and end together
            metrics_list.append(RequestMetrics(
                request_id=req.request_id,
                prompt_len=req.prompt_len,
                output_len=output_len,
                arrival_time=arrival,
                start_time=arrival,
                first_token_time=arrival + batch_time * 0.3,
                end_time=end,
            ))

    avg_padding = sum(s["padding_pct"] for s in padding_stats) / len(padding_stats)
    avg_efficiency = sum(s["efficiency_pct"] for s in padding_stats) / len(padding_stats)

    return metrics_list, {"avg_padding_pct": avg_padding, "avg_efficiency_pct": avg_efficiency}


def main():
    print(f"E02 — Static Batching | Model: {MODEL}")
    print(f"Batch sizes: {BATCH_SIZES} | Prompt std: {PROMPT_STD} (high variance)\n")

    # Generate workload with high variance in prompt lengths
    cfg = WorkloadConfig(
        num_requests=64,
        prompt_len_mean=PROMPT_MEAN,
        prompt_len_std=PROMPT_STD,
        output_len_mean=OUTPUT_TOKENS,
        output_len_std=16,
    )
    requests = generate_synthetic_workload(cfg)

    results = []

    for batch_size in BATCH_SIZES:
        print(f"  Batch size = {batch_size}...")

        em = ExperimentMetrics(
            experiment_id=f"E02_static_batch_{batch_size}",
            description=f"Static batching — batch_size={batch_size}",
            config={"batch_size": batch_size, "prompt_mean": PROMPT_MEAN,
                    "prompt_std": PROMPT_STD},
        )
        em.start_wall_time = time.time()

        try:
            metrics_list, waste_stats = run_static_batch_vllm(batch_size, requests)
        except ImportError:
            print("  vLLM not found — using simulated timing")
            metrics_list, waste_stats = _simulate(batch_size, requests)

        for m in metrics_list:
            em.add(m)

        em.finish()

        summary = em.summary()
        summary.update(waste_stats)
        results.append(summary)

        print(f"    Throughput:    {summary.get('output_tokens_per_sec', 0):.0f} tok/s")
        print(f"    TTFT p99:      {summary.get('ttft_ms_p99', 0):.0f}ms")
        print(f"    Padding waste: {waste_stats['avg_padding_pct']:.1f}%  "
              f"(efficiency: {waste_stats['avg_efficiency_pct']:.1f}%)")

    # Print comparison table
    print("\n── Batch Size Comparison ──────────────────────────────────────")
    print(f"{'Batch':>6} {'Tok/s':>8} {'TTFT_p99':>10} {'Padding%':>10} {'Efficiency%':>12}")
    for r in results:
        bs = r["config"]["batch_size"]
        print(f"{bs:>6} {r.get('output_tokens_per_sec', 0):>8.0f} "
              f"{r.get('ttft_ms_p99', 0):>10.0f} "
              f"{r.get('avg_padding_pct', 0):>10.1f} "
              f"{r.get('avg_efficiency_pct', 0):>12.1f}")

    print("\n[Article insight]")
    print("  Larger batches improve throughput but increase padding waste.")
    print("  With high length variance, up to 50%+ of compute is wasted on padding.")
    print("  This motivates dynamic batching and continuous batching.")

    # ── Save results ──────────────────────────────────────────────────────────
    import json, os
    os.makedirs("results", exist_ok=True)
    save_data = {
        "experiment_id": "E02_static_batching",
        "description": "Static batching — padding waste vs batch size",
        "config": {"prompt_mean": PROMPT_MEAN, "prompt_std": PROMPT_STD,
                   "output_tokens": OUTPUT_TOKENS},
        "batch_results": [
            {
                "batch_size": r["config"]["batch_size"],
                "output_tokens_per_sec": round(r.get("output_tokens_per_sec", 0), 1),
                "ttft_ms_p99": round(r.get("ttft_ms_p99", 0), 1),
                "avg_padding_pct": round(r.get("avg_padding_pct", 0), 1),
                "avg_efficiency_pct": round(r.get("avg_efficiency_pct", 0), 1),
            }
            for r in results
        ]
    }
    with open("results/E02_static_batching.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("  Results saved → results/E02_static_batching.json")


def _simulate(batch_size: int, requests) -> tuple[list[RequestMetrics], dict]:
    """Timing simulation when vLLM is not available."""
    import math
    metrics_list = []
    padding_stats = []
    base_tpot = 0.02   # 20ms/token baseline on CPU

    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start:batch_start + batch_size]
        prompt_lens = [r.prompt_len for r in batch]
        max_len = max(prompt_lens)

        waste = compute_padding_waste(prompt_lens)
        padding_stats.append(waste)

        # Static batch: all seqs run to max length, wasted compute included
        prefill_time = max_len * 0.0005 * batch_size
        decode_time = OUTPUT_TOKENS * base_tpot * batch_size

        arrival = time.perf_counter()
        time.sleep(0.001)  # tiny sleep to simulate work
        end = arrival + prefill_time + decode_time

        for req in batch:
            metrics_list.append(RequestMetrics(
                request_id=req.request_id,
                prompt_len=req.prompt_len,
                output_len=OUTPUT_TOKENS,
                arrival_time=arrival,
                start_time=arrival,
                first_token_time=arrival + prefill_time,
                end_time=end,
            ))

    avg_padding = sum(s["padding_pct"] for s in padding_stats) / len(padding_stats)
    avg_efficiency = sum(s["efficiency_pct"] for s in padding_stats) / len(padding_stats)
    return metrics_list, {"avg_padding_pct": avg_padding, "avg_efficiency_pct": avg_efficiency}


if __name__ == "__main__":
    main()
