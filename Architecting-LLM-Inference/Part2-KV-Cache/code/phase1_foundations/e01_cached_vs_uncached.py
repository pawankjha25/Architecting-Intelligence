"""
E01 -- Cached vs. Uncached Decoding
=====================================
Goal: Show the TWO separate stories that get conflated in casual explanations
      of KV caching:

  (A) NO CACHE -> a COMPUTE / LATENCY problem, not a memory-growth problem.
      Each decode step recomputes attention over the full sequence so far
      (O(n^2) total across a generation). Nothing persists between steps --
      the transient activation memory is allocated and freed every step.
      The story here is tokens/sec collapsing and per-token latency growing
      as generation gets longer, NOT standing memory growth.

  (B) NAIVE UNBOUNDED PERSISTENT CACHE, MANY CONCURRENT LONG SEQUENCES ->
      a MEMORY problem. This is where GPU memory actually grows monotonically
      and eventually OOMs, because nothing is ever evicted.

Phase A (this file, MacBook / CPU, no GPU needed): pure math/timing simulation
         of the two curves using FLOP-proportional dummy work, so the shapes
         are correct even without a real model loaded.
Phase B (optional, requires `transformers` + a small model, ideally with a
         real GPU to see the memory numbers move): swap in
         `run_with_real_model()` below on your machine.

Run: python3 phase1_foundations/e01_cached_vs_uncached.py
"""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.memory_model import MODEL_CONFIGS, kv_cache_bytes, human_bytes


# -- Phase A: FLOP-proportional simulation (no GPU / no model needed) ----------

def simulate_uncached_step_cost(position: int) -> float:
    """
    Without a cache, decoding token at `position` requires attention over
    all `position` prior tokens -- cost grows linearly per step, so total
    cost across a generation of length N grows as O(N^2).
    We return a synthetic "compute units" cost, not wall-clock time, since
    this file has no GPU/model -- the point is the *shape* of the curve.
    """
    return position  # attention cost scales with current sequence length


def simulate_cached_step_cost(position: int) -> float:
    """
    With a KV cache, each decode step only computes attention for the ONE
    new token against the cached K/V of all prior tokens -- the QK^T/softmax/V
    matmuls are O(position) in memory *reads* but the newly computed work per
    step is O(1) relative to recomputing the whole prefix from scratch.
    We model this as a constant unit cost per step (the realistic behavior:
    TPOT stays roughly flat as generation gets longer, aside from memory
    bandwidth effects we ignore here for clarity).
    """
    return 1.0


def run_compute_curves(num_steps: int = 512) -> dict:
    uncached_costs = [simulate_uncached_step_cost(p) for p in range(1, num_steps + 1)]
    cached_costs = [simulate_cached_step_cost(p) for p in range(1, num_steps + 1)]

    uncached_cumulative = sum(uncached_costs)
    cached_cumulative = sum(cached_costs)

    return {
        "num_steps": num_steps,
        "uncached_total_compute_units": uncached_cumulative,
        "cached_total_compute_units": cached_cumulative,
        "blowup_factor": uncached_cumulative / cached_cumulative,
        "uncached_step_100": uncached_costs[99] if num_steps >= 100 else None,
        "cached_step_100": cached_costs[99] if num_steps >= 100 else None,
        "uncached_last_step": uncached_costs[-1],
        "cached_last_step": cached_costs[-1],
    }


def run_memory_curve_naive_persistent_cache(
    model_key: str = "llama3-8b",
    seq_len: int = 4096,
    max_concurrent_requests: int = 64,
    gpu_memory_budget_bytes: float = 24 * 1024 ** 3,   # 24GB, e.g. RTX 3090
) -> dict:
    """
    (B) The actual memory-growth story: a naive cache manager that never
    evicts, serving an increasing number of concurrent long sequences.
    """
    cfg = MODEL_CONFIGS[model_key]
    curve = []
    cumulative_bytes = 0.0
    oom_at = None

    for n in range(1, max_concurrent_requests + 1):
        per_seq_bytes = kv_cache_bytes(cfg, seq_len=seq_len, batch_size=1, dtype="fp16")
        cumulative_bytes += per_seq_bytes
        curve.append({"concurrent_requests": n, "cumulative_kv_bytes": cumulative_bytes})
        if cumulative_bytes > gpu_memory_budget_bytes and oom_at is None:
            oom_at = n

    return {
        "model": model_key,
        "seq_len": seq_len,
        "gpu_memory_budget": human_bytes(gpu_memory_budget_bytes),
        "oom_at_concurrent_requests": oom_at,
        "curve_sample": curve[::8],   # every 8th point to keep output short
    }


# -- Phase B: real model, requires `transformers` (and ideally a GPU) ----------

def run_with_real_model(model_name: str = "gpt2", num_new_tokens: int = 64):
    """
    Optional: run this on a machine with `pip install torch transformers`.
    Compares wall-clock time and (if CUDA available) memory for
    use_cache=True vs use_cache=False.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    inputs = tok(prompt, return_tensors="pt").to(device)

    results = {}
    for use_cache in (True, False):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=num_new_tokens,
                use_cache=use_cache,
                do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else None
        results[f"use_cache={use_cache}"] = {
            "elapsed_s": elapsed,
            "tokens_per_sec": num_new_tokens / elapsed,
            "peak_memory_bytes": peak_mem,
        }

    return results


def main():
    print("E01 -- Cached vs. Uncached Decoding\n")

    print("-- (A) Compute/latency blowup without a cache --------------------")
    compute = run_compute_curves(num_steps=512)
    print(f"  Over {compute['num_steps']} decode steps:")
    print(f"  No cache:   total compute units = {compute['uncached_total_compute_units']:.0f}  "
          f"(step 512 alone costs {compute['uncached_last_step']:.0f} units)")
    print(f"  With cache: total compute units = {compute['cached_total_compute_units']:.0f}  "
          f"(step 512 alone costs {compute['cached_last_step']:.0f} unit)")
    print(f"  -> {compute['blowup_factor']:.1f}x more total compute without caching "
          f"(grows quadratically, O(n^2), vs. linearly, O(n), with caching)")

    print("\n-- (B) Memory blowup: naive unbounded persistent cache ------------")
    mem = run_memory_curve_naive_persistent_cache()
    print(f"  Model: {mem['model']}, seq_len={mem['seq_len']}, "
          f"GPU budget={mem['gpu_memory_budget']}")
    if mem["oom_at_concurrent_requests"]:
        print(f"  -> OOM at {mem['oom_at_concurrent_requests']} concurrent requests "
              f"if nothing is ever evicted.")
    else:
        print("  -> Did not OOM within the tested range; raise max_concurrent_requests "
              "or seq_len to see it.")
    for point in mem["curve_sample"]:
        from benchmarks.memory_model import human_bytes as hb
        print(f"    {point['concurrent_requests']:3d} requests -> "
              f"{hb(point['cumulative_kv_bytes'])}")

    print("\n[Article insight]")
    print("  These are two DIFFERENT failure modes, not one:")
    print("  - No cache => compute/latency explodes (nothing persists in memory).")
    print("  - Naive persistent cache + no eviction + many long sequences => memory explodes.")
    print("  Confusing the two leads to the wrong fix (e.g. 'add more memory' when the")
    print("  real problem is recomputation cost, or vice versa).")

    Path("results").mkdir(exist_ok=True)
    with open("results/E01_cached_vs_uncached.json", "w") as f:
        json.dump({"compute": compute, "memory": mem}, f, indent=2)
    print("\n  Results saved -> results/E01_cached_vs_uncached.json")


if __name__ == "__main__":
    main()
