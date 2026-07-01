"""
E06 (Phase B) -- Full Cache vs. Sliding Window, real model
==============================================================
Replaces E06's synthetic needle-survival proxy with a real measurement:
perplexity on real text, under a real sliding-window cache policy applied
during a teacher-forced continuation, vs. an unbounded full cache.

Run on: RunPod GPU (RTX 3090+). Requires torch, transformers, CUDA.
Model: Qwen/Qwen2.5-0.5B-Instruct (default; override via --model)

NOTE: this file was authored without GPU access and has not been executed.
The logic in gpu_cache_utils.py was written carefully against the
DynamicCache API, but if you hit an error, it's most likely an API version
mismatch -- see the version-sensitivity note at the top of that file.

Run: python3 phase3_footprint_opt/e06_sliding_window_gpu.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.gpu_cache_utils import (
    DEFAULT_MODEL, gpu_available, load_model_and_tokenizer,
    perplexity_under_policy, make_sliding_window_policy, load_long_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    print("E06 (Phase B) -- Full Cache vs. Sliding Window (real model)\n")

    if not gpu_available():
        print("  No CUDA GPU detected. Run this on your RunPod pod:")
        print("    pip install -r requirements-runpod.txt torch transformers")
        print(f"    python3 {__file__.split('/')[-1]}")
        return

    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")
    text = load_long_text(tokenizer, min_tokens=args.warmup_tokens + args.max_tokens + 100)

    window_sizes = [64, 128, 256, 512]
    print(f"Model: {args.model}, warmup={args.warmup_tokens}, measured_tokens={args.max_tokens}\n")

    print("-- Full cache (no eviction) baseline -------------------------------")
    full = perplexity_under_policy(
        model, tokenizer, text, policy_fn=None,
        warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
    )
    print(f"  perplexity={full.perplexity:.3f}  avg_nll={full.avg_nll:.4f}  "
          f"final_cache_len={full.final_cache_len}")

    print("\n-- Sliding window sweep --------------------------------------------")
    print(f"{'window':>8} | {'perplexity':>12} | {'avg_nll':>10} | {'final_cache_len':>16}")
    print("-" * 55)
    rows = [{"window": "full_cache", "perplexity": full.perplexity, "avg_nll": full.avg_nll,
             "final_cache_len": full.final_cache_len}]
    for w in window_sizes:
        policy = make_sliding_window_policy(w)
        policy.__name__ = f"sliding_window_{w}"
        result = perplexity_under_policy(
            model, tokenizer, text, policy_fn=policy,
            warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
        )
        rows.append({"window": w, "perplexity": result.perplexity, "avg_nll": result.avg_nll,
                     "final_cache_len": result.final_cache_len})
        print(f"{w:>8} | {result.perplexity:>12.3f} | {result.avg_nll:>10.4f} | {result.final_cache_len:>16}")

    print("\n[Article insight]")
    print("  Perplexity should rise as window shrinks -- the real-model counterpart to")
    print("  E06's synthetic needle-survival curve. Unlike the proxy, this reflects an")
    print("  actual model's actual next-token predictions degrading as it loses access")
    print("  to earlier context, on the specific text used here. Swap in your own long")
    print("  document via `load_long_text` for numbers representative of your workload --")
    print("  the built-in repeated passage is convenient but not a realistic corpus.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E06_sliding_window_gpu.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  Results saved -> results/E06_sliding_window_gpu.json")


if __name__ == "__main__":
    main()
