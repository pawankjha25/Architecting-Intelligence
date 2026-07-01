"""
E08 (Phase B) -- Recency vs. Real Attention-Based (H2O-style) Eviction
==========================================================================
Replaces E08's synthetic importance-score proxy with real attention scores
captured via `output_attentions=True` during generation, accumulated across
steps, and used to decide retention -- the actual H2O mechanism, not a
stand-in for it.

Run on: RunPod GPU (RTX 3090+). Requires torch, transformers, CUDA.
NOTE: authored without GPU access; not executed before commit.
`output_attentions=True` adds real memory/compute overhead per step --
expect this to run noticeably slower than E06B/E07B. If you hit OOM, lower
--max-tokens or use an even smaller model.

Run: python3 phase3_footprint_opt/e08_eviction_policies_gpu.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.gpu_cache_utils import (
    DEFAULT_MODEL, gpu_available, load_model_and_tokenizer,
    perplexity_under_policy, make_sliding_window_policy, make_heavy_hitter_policy,
    load_long_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cache-budget", type=int, default=128)
    args = parser.parse_args()

    print("E08 (Phase B) -- Recency vs. Real Attention-Based Eviction (real model)\n")

    if not gpu_available():
        print("  No CUDA GPU detected. Run this on your RunPod pod:")
        print("    pip install -r requirements-runpod.txt torch transformers")
        print(f"    python3 {__file__.split('/')[-1]}")
        return

    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")
    text = load_long_text(tokenizer, min_tokens=args.warmup_tokens + args.max_tokens + 100)

    print(f"Model: {args.model}, cache_budget={args.cache_budget}\n")

    print("-- FIFO / sliding window (recency only) -----------------------------")
    fifo_policy = make_sliding_window_policy(args.cache_budget)
    fifo_policy.__name__ = "fifo"
    fifo_result = perplexity_under_policy(
        model, tokenizer, text, policy_fn=fifo_policy,
        warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
    )
    print(f"  perplexity={fifo_result.perplexity:.3f}  avg_nll={fifo_result.avg_nll:.4f}")

    print("\n-- H2O-style heavy-hitter (real attention scores) -------------------")
    hh_policy = make_heavy_hitter_policy(cache_budget=args.cache_budget, recent_protect=args.cache_budget // 4)
    hh_policy.__name__ = "heavy_hitter"
    hh_result = perplexity_under_policy(
        model, tokenizer, text, policy_fn=hh_policy,
        warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
        needs_attentions=True,
    )
    print(f"  perplexity={hh_result.perplexity:.3f}  avg_nll={hh_result.avg_nll:.4f}")

    improvement_pct = (fifo_result.perplexity - hh_result.perplexity) / fifo_result.perplexity * 100

    print(f"\n  Improvement: {improvement_pct:+.1f}% perplexity reduction from heavy-hitter "
          f"retention over pure recency at the same cache_budget={args.cache_budget}")

    print("\n[Article insight]")
    print("  Unlike E08's synthetic needle test (which engineered a scenario where the")
    print("  needle's importance was artificially boosted), this measures whatever")
    print("  actual importance structure the model's own attention has on real text.")
    print("  A real improvement here means the model itself concentrates attention on")
    print("  a non-recency-correlated subset of tokens often enough that tracking")
    print("  cumulative attention score genuinely beats pure recency -- if the")
    print("  improvement is small or negative, that's a legitimate finding too: it")
    print("  would mean this model's real attention patterns are recency-dominated on")
    print("  this text, and the synthetic E08 result was demonstrating the mechanism,")
    print("  not asserting it always wins by a specific margin.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E08_eviction_policies_gpu.json", "w") as f:
        json.dump({
            "cache_budget": args.cache_budget,
            "fifo": {"perplexity": fifo_result.perplexity, "avg_nll": fifo_result.avg_nll},
            "heavy_hitter": {"perplexity": hh_result.perplexity, "avg_nll": hh_result.avg_nll},
            "improvement_pct": improvement_pct,
        }, f, indent=2)
    print("\n  Results saved -> results/E08_eviction_policies_gpu.json")


if __name__ == "__main__":
    main()
