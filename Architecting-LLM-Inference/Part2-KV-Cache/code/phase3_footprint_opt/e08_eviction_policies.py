"""
E08 -- Recency vs. Attention-Based Eviction
==============================================
Goal: Show the case where H2O-style heavy-hitter retention actually wins:
      a token that mattered EARLIER in generation (so it accumulated a high
      importance score) but is no longer recent. Pure recency policies
      (FIFO / sliding window / sink+window) all lose it once it ages out;
      heavy-hitter retention keeps it because it looks at cumulative
      importance, not just position.

SYNTHETIC PROXY: "importance score" is synthetically assigned (simulating
"this token was heavily attended to earlier"), not measured from a real
model. See retention_sim.py docstring for the full honesty note.

Run: python3 phase3_footprint_opt/e08_eviction_policies.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.retention_sim import (
    run_needle_test, fifo_retained, attention_sink_retained, heavy_hitter_retained,
)


def main():
    print("E08 -- Recency vs. Attention-Based Eviction\n")

    seq_len = 8192
    cache_budget = 1024

    print("-- Scenario 1: needle has NO special importance (looks like any other token) --")
    print("  (the needle is just some middle-of-sequence content, never specially attended to)\n")
    results_plain = {}
    results_plain["FIFO / sliding window"] = run_needle_test(fifo_retained, seq_len, cache_budget, num_trials=1000)
    results_plain["Sink + window"] = run_needle_test(attention_sink_retained, seq_len, cache_budget, num_trials=1000, num_sinks=4)
    results_plain["H2O-style heavy-hitter"] = run_needle_test(
        heavy_hitter_retained, seq_len, cache_budget, num_trials=1000, needle_score_boost=0.0,
    )
    for name, rate in results_plain.items():
        print(f"  {name:26s}  {rate:.1%}")

    print("\n-- Scenario 2: needle WAS heavily attended to earlier (high historical importance) --")
    print("  (simulates: this token got referenced a lot during generation, then generation moved on)\n")
    results_boosted = {}
    results_boosted["FIFO / sliding window"] = run_needle_test(fifo_retained, seq_len, cache_budget, num_trials=1000)
    results_boosted["Sink + window"] = run_needle_test(attention_sink_retained, seq_len, cache_budget, num_trials=1000, num_sinks=4)
    results_boosted["H2O-style heavy-hitter"] = run_needle_test(
        heavy_hitter_retained, seq_len, cache_budget, num_trials=1000, needle_score_boost=5.0,
    )
    for name, rate in results_boosted.items():
        print(f"  {name:26s}  {rate:.1%}")

    print("\n[Article insight]")
    print("  In Scenario 1, all three policies perform similarly (roughly cache_budget/")
    print("  seq_len for the recency-based ones, and H2O with no signal to work from")
    print("  reduces to something similar). The interesting result is Scenario 2: once")
    print("  the needle carries a real importance signal, H2O's cumulative-score")
    print("  retention keeps it reliably, while FIFO/sliding-window and sink+window")
    print("  still evict it purely because it's not recent -- they have no mechanism")
    print("  to know it mattered. This is H2O's actual value proposition: it isn't")
    print("  better at everything, it specifically rescues important-but-not-recent")
    print("  tokens that pure recency policies are structurally blind to.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E08_eviction_policies.json", "w") as f:
        json.dump({"no_signal": results_plain, "with_importance_signal": results_boosted}, f, indent=2)
    print("\n  Results saved -> results/E08_eviction_policies.json")


if __name__ == "__main__":
    main()
