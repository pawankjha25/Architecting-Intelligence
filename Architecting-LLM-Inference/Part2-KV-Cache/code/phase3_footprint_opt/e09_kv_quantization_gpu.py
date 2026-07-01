"""
E09 (Phase B) -- KV Precision Frontier, real captured tensors
=================================================================
Replaces E09's synthetic Gaussian-plus-outliers KV distribution with the
real thing: actual K/V tensors captured during a real forward pass on real
text, fake-quantized (round-tripped through reduced precision) at several
bit widths, with the perplexity impact measured directly rather than via
reconstruction MSE on synthetic data.

Run on: RunPod GPU (RTX 3090+). Requires torch, transformers, CUDA.
NOTE: authored without GPU access; not executed before commit.

Run: python3 phase3_footprint_opt/e09_kv_quantization_gpu.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_footprint_opt.gpu_cache_utils import (
    DEFAULT_MODEL, gpu_available, load_model_and_tokenizer,
    perplexity_under_policy, make_quantization_policy, load_long_text,
)
from benchmarks.memory_model import DTYPE_BYTES, human_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    print("E09 (Phase B) -- KV Precision Frontier (real captured tensors)\n")

    if not gpu_available():
        print("  No CUDA GPU detected. Run this on your RunPod pod:")
        print("    pip install -r requirements-runpod.txt torch transformers")
        print(f"    python3 {__file__.split('/')[-1]}")
        return

    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")
    text = load_long_text(tokenizer, min_tokens=args.warmup_tokens + args.max_tokens + 100)

    print(f"Model: {args.model}\n")

    print("-- bf16 baseline (no quantization) ----------------------------------")
    baseline = perplexity_under_policy(
        model, tokenizer, text, policy_fn=None,
        warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
    )
    print(f"  perplexity={baseline.perplexity:.3f}  avg_nll={baseline.avg_nll:.4f}")

    print("\n-- Fake-quantized sweep (real K/V tensors, every decode step) -------")
    print(f"{'bits':>6} | {'bytes/elem':>10} | {'perplexity':>12} | {'delta_vs_bf16':>14}")
    print("-" * 52)

    rows = [{"bits": 16, "perplexity": baseline.perplexity, "avg_nll": baseline.avg_nll}]
    for bits, dtype_key in ((8, "int8"), (4, "int4"), (2, "int2")):
        policy = make_quantization_policy(bits=bits, every_n_steps=1)
        policy.__name__ = f"quantize_{bits}bit"
        result = perplexity_under_policy(
            model, tokenizer, text, policy_fn=policy,
            warmup_tokens=args.warmup_tokens, max_tokens=args.max_tokens,
        )
        delta_pct = (result.perplexity - baseline.perplexity) / baseline.perplexity * 100
        rows.append({"bits": bits, "perplexity": result.perplexity, "avg_nll": result.avg_nll,
                     "delta_vs_bf16_pct": delta_pct})
        print(f"{bits:>6} | {DTYPE_BYTES[dtype_key]:>10} | {result.perplexity:>12.3f} | {delta_pct:>+13.1f}%")

    print("\n[Article insight]")
    print("  This fake-quantization is a naive min-max scheme applied uniformly --")
    print("  no per-channel (keys) / per-token (values) asymmetry, no outlier")
    print("  isolation, unlike KIVI or KVQuant. Expect it to look worse than published")
    print("  numbers for those methods at the same bit width -- that gap IS the point:")
    print("  it's what a real per-channel/per-token/outlier-aware scheme buys you over")
    print("  naive quantization, measured on this model rather than assumed from the")
    print("  papers. If int8 shows negligible perplexity change here, that matches the")
    print("  common finding that int8 KV cache is close to free; if int4/int2 degrade")
    print("  sharply, that's consistent with needing the more careful schemes below")
    print("  4 bits.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E09_kv_quantization_gpu.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  Results saved -> results/E09_kv_quantization_gpu.json")


if __name__ == "__main__":
    main()
