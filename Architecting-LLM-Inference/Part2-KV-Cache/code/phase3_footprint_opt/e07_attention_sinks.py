"""
E07 -- Sliding Window vs. Attention Sinks
============================================
Goal: Test the actual StreamingLLM claim correctly. The paper's finding is
      about GENERATION STABILITY, not improved distant recall: dropping the
      first few tokens causes attention to destabilize (the "attention sink"
      phenomenon), and pinning them fixes that -- it does NOT give you back
      the ability to recall arbitrary distant content.

SYNTHETIC PROXY: no real model is loaded. We model two things:
  1. A "collapse score" -- a synthetic stand-in for the real observation
     that output degrades once the first few tokens are evicted. Modeled
     as a step function: collapse is high when sink tokens are missing,
     low when they're present. This is illustrative, not measured.
  2. Needle survival for a needle placed in the MIDDLE of the sequence --
     this should show attention sinks give almost NO benefit over a plain
     window here, because sinks only preserve the first few tokens, not
     arbitrary middle content. That result is the honest, correct one.

Run: python3 phase3_footprint_opt/e07_attention_sinks.py
"""

import sys
import os
import json
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.retention_sim import run_needle_test, sliding_window_retained, attention_sink_retained


def synthetic_collapse_score(num_sinks_retained: int) -> float:
    """
    Illustrative proxy for the StreamingLLM finding: collapse is severe
    when zero initial tokens are retained, and resolves once even a small
    number of sink tokens are pinned. Not a measured quantity.
    """
    if num_sinks_retained == 0:
        return 0.85   # high synthetic "collapse" score
    return max(0.05, 0.3 / (num_sinks_retained + 1))


def main():
    print("E07 -- Sliding Window vs. Attention Sinks\n")

    seq_len = 8192
    cache_budget = 1024
    num_sinks_options = [0, 1, 2, 4, 8, 16]

    print("-- (1) Synthetic 'collapse' proxy vs. number of retained sink tokens --")
    print(f"{'num_sinks':>10} | {'collapse score (proxy, lower=better)':>38}")
    print("-" * 55)
    collapse_rows = []
    for n in num_sinks_options:
        score = synthetic_collapse_score(n)
        collapse_rows.append({"num_sinks": n, "collapse_score_proxy": score})
        print(f"{n:>10} | {score:>38.2f}")

    print("\n-- (2) Needle-in-the-middle survival: plain window vs. window+sinks ----")
    mid_needle_survival_window = run_needle_test(
        sliding_window_retained, seq_len, cache_budget, num_trials=1000,
    )
    mid_needle_survival_sink = run_needle_test(
        attention_sink_retained, seq_len, cache_budget, num_trials=1000, num_sinks=4,
    )
    print(f"  Plain sliding window:        {mid_needle_survival_window:.1%}")
    print(f"  Window + 4 attention sinks:  {mid_needle_survival_sink:.1%}")
    print(f"  Difference: {(mid_needle_survival_sink - mid_needle_survival_window)*100:+.2f} pp "
          f"(small -- sinks help stability, not distant middle-content recall)")

    print("\n[Article insight]")
    print("  These two results tell different, complementary stories, and it's easy to")
    print("  conflate them: (1) shows why you must never evict ALL initial tokens --")
    print("  the synthetic collapse proxy spikes at zero retained sinks, matching the")
    print("  real StreamingLLM finding that generation destabilizes without them. (2)")
    print("  shows sinks barely move the needle (pun intended) on recalling arbitrary")
    print("  middle-of-sequence content -- they only protect the first few positions,")
    print("  not everything that might matter. Sinks are a stability fix, not a recall")
    print("  fix; H2O-style heavy-hitter retention (E08) is what targets recall.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E07_attention_sinks.json", "w") as f:
        json.dump({
            "collapse_proxy": collapse_rows,
            "needle_middle_survival": {
                "plain_window": mid_needle_survival_window,
                "window_plus_sinks": mid_needle_survival_sink,
            },
        }, f, indent=2)
    print("\n  Results saved -> results/E07_attention_sinks.json")


if __name__ == "__main__":
    main()
