"""
E06 -- Full Cache vs. Sliding Window
=======================================
Goal: Quantify the memory/concurrency win from a sliding window against the
      retrieval capability it gives up.

SYNTHETIC PROXY (see retention_sim.py docstring for full honesty note): no
real model is loaded here. "Needle survival rate" stands in for real
long-context retrieval accuracy, and tests the RETENTION POLICY's behavior,
not any specific model's attention.

Run: python3 phase3_footprint_opt/e06_sliding_window.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.memory_model import MODEL_CONFIGS, kv_cache_bytes, human_bytes
from phase3_footprint_opt.retention_sim import run_needle_test, sliding_window_retained


def full_cache_retained(seq_len, cache_budget, position):
    return True  # full cache never evicts


def main():
    print("E06 -- Full Cache vs. Sliding Window\n")

    seq_len = 8192
    window_sizes = [256, 512, 1024, 2048, 4096, 8192]  # 8192 == full cache
    model_key = "qwen2.5-7b"
    cfg = MODEL_CONFIGS[model_key]

    print(f"Sequence length: {seq_len}, model: {model_key}\n")
    print(f"{'window':>8} | {'memory':>10} | {'reduction':>10} | {'needle survival (proxy)':>24}")
    print("-" * 62)

    rows = []
    for w in window_sizes:
        mem = kv_cache_bytes(cfg, seq_len=w, batch_size=1, dtype="fp16")
        full_mem = kv_cache_bytes(cfg, seq_len=seq_len, batch_size=1, dtype="fp16")
        reduction_pct = (1 - mem / full_mem) * 100

        if w >= seq_len:
            survival = run_needle_test(full_cache_retained, seq_len, w, num_trials=500)
        else:
            survival = run_needle_test(sliding_window_retained, seq_len, w, num_trials=500)

        rows.append({
            "window": w, "memory_bytes": mem, "reduction_pct": reduction_pct,
            "needle_survival_rate": survival,
        })
        print(f"{w:>8} | {human_bytes(mem):>10} | {reduction_pct:>9.1f}% | {survival:>23.1%}")

    print("\n[Article insight]")
    print("  Memory scales down linearly with window size, as expected. The needle-")
    print("  survival proxy shows something sharper: survival rate tracks window/seq_len")
    print("  almost exactly, because a plain sliding window retains information purely")
    print("  by recency -- anything outside the window is gone with no chance of")
    print("  recall, regardless of how important it was. This is the real tradeoff:")
    print("  sliding window buys a hard memory bound at the cost of a hard recall")
    print("  cutoff, not a graceful quality degradation.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E06_sliding_window.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  Results saved -> results/E06_sliding_window.json")


if __name__ == "__main__":
    main()
