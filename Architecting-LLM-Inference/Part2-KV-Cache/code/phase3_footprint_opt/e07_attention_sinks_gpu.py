"""
E07 (Phase B) -- Sliding Window vs. Attention Sinks, real model
===================================================================
Replaces E07's synthetic "collapse score" with a real measurement: does
perplexity actually spike when zero initial tokens are retained, and does
pinning a handful of sink tokens actually fix it? This is the StreamingLLM
claim, tested directly rather than via a synthetic proxy.

Run on: RunPod GPU (RTX 3090+). Requires torch, transformers, CUDA.
NOTE: authored without GPU access; not executed before commit. See
gpu_cache_utils.py's version-sensitivity note if something breaks.

Run: python3 phase3_footprint_opt/e07_attention_sinks_gpu.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.gpu_cache_utils import (
    DEFAULT_MODEL, gpu_available, load_model_and_tokenizer,
    perplexity_under_policy, make_sliding_window_policy, make_attention_sink_policy,
    load_long_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--window", type=int, default=128,
                         help="small window on purpose -- this is where the collapse effect should be visible")
    args = parser.parse_args()

    print("E07 (Phase B) -- Sliding Window vs. Attention Sinks (real model)\n")

    if not gpu_available():
        print("  No CUDA GPU detected. Run this on your RunPod pod:")
        print("    pip install -r requirements-runpod.txt torch transformers")
        print(f"    python3 {__file__.split('/')[-1]}")
        return

    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")
    text = load_long_text(tokenizer, min_tokens=args.warmup_tokens + args.max_tokens + 100)

    print(f"Model: {args.model}, window={args.window}\n")

    print("-- (1) Plain sliding window, NO sinks retained ----------------------")
    plain_window = make_sliding_window_policy(args.window)
    plain_window.__name__ = "plain_sliding_window"
    plain_result = perplexity_under_policy(
        model, tokenizer, text, policy_fn=plain_window,
        warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
    )
    print(f"  perplexity={plain_result.perplexity:.3f}  avg_nll={plain_result.avg_nll:.4f}")

    print("\n-- (2) Window + attention sinks sweep -------------------------------")
    print(f"{'num_sinks':>10} | {'perplexity':>12} | {'avg_nll':>10}")
    print("-" * 40)
    rows = [{"num_sinks": 0, "perplexity": plain_result.perplexity, "avg_nll": plain_result.avg_nll}]
    for num_sinks in (1, 2, 4, 8, 16):
        policy = make_attention_sink_policy(num_sinks=num_sinks, window_size=args.window)
        policy.__name__ = f"attention_sink_{num_sinks}"
        result = perplexity_under_policy(
            model, tokenizer, text, policy_fn=policy,
            warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
        )
        rows.append({"num_sinks": num_sinks, "perplexity": result.perplexity, "avg_nll": result.avg_nll})
        print(f"{num_sinks:>10} | {result.perplexity:>12.3f} | {result.avg_nll:>10.4f}")

    print("\n[Article insight]")
    print("  If the StreamingLLM effect holds on this model, perplexity at num_sinks=0")
    print("  should be visibly worse than at num_sinks>=1, with most of the benefit")
    print("  captured by just a handful of sinks (diminishing returns beyond ~4). If")
    print("  the effect is small or absent here, that's a real and worth-reporting")
    print("  result too -- the original StreamingLLM findings were demonstrated on")
    print("  specific model families; not every architecture necessarily exhibits the")
    print("  same sink dependency. Report what you actually measure, not what the")
    print("  paper says should happen.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E07_attention_sinks_gpu.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  Results saved -> results/E07_attention_sinks_gpu.json")


if __name__ == "__main__":
    main()
