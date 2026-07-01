"""
E09 -- KV Precision Frontier
===============================
Goal: Quantify the memory-vs-reconstruction-error tradeoff of quantizing KV
      cache values at decreasing bit widths.

SYNTHETIC PROXY: no real model is loaded, so K/V values are drawn from a
distribution that approximates what's reported in the literature (roughly
Gaussian per-channel, with occasional outlier channels -- this is why real
quantization schemes like KIVI use asymmetric per-channel (keys) vs
per-token (values) scaling rather than one flat scheme). We measure
reconstruction error (MSE, SNR) as a PROXY for downstream quality
degradation, not real perplexity or task accuracy -- those require an
actual model and are out of scope without GPU/model access.

Phase B (real model, requires transformers + GPU): swap `synthetic_kv_values`
for real K/V tensors captured from a forward pass, and measure actual
downstream perplexity / task accuracy instead of reconstruction MSE.

Run: python3 phase3_footprint_opt/e09_kv_quantization.py
"""

import sys
import os
import json
import random
import math
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.memory_model import DTYPE_BYTES, human_bytes


def synthetic_kv_values(n: int, num_outlier_channels: int = 4, seed: int = 42) -> list:
    """
    Approximate real KV value distributions: mostly Gaussian, with a small
    number of "outlier channel" values drawn from a wider distribution --
    this is the reason naive uniform quantization struggles and per-channel/
    per-token schemes (KIVI, KVQuant) exist in the first place.
    """
    rng = random.Random(seed)
    values = [rng.gauss(0, 1.0) for _ in range(n)]
    for _ in range(num_outlier_channels):
        idx = rng.randint(0, n - 1)
        values[idx] = rng.gauss(0, 8.0)   # outlier magnitude
    return values


def quantize_dequantize(values: list, bits: int) -> list:
    """Simple symmetric min-max scalar quantization (illustrative, not KIVI's actual scheme)."""
    if bits >= 16:
        return values[:]  # no-op for bf16/fp16 baseline
    levels = 2 ** bits
    v_min, v_max = min(values), max(values)
    scale = (v_max - v_min) / max(levels - 1, 1)
    if scale == 0:
        return values[:]
    quantized = [round((v - v_min) / scale) for v in values]
    dequantized = [q * scale + v_min for q in quantized]
    return dequantized


def mse(a: list, b: list) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def snr_db(original: list, reconstructed: list) -> float:
    signal_power = sum(x ** 2 for x in original) / len(original)
    noise_power = mse(original, reconstructed)
    if noise_power == 0:
        return float("inf")
    return 10 * math.log10(signal_power / noise_power)


def main():
    print("E09 -- KV Precision Frontier\n")

    n_values = 100_000
    original = synthetic_kv_values(n_values)

    configs = [
        ("bf16/fp16", 16),
        ("fp8", 8),
        ("int8", 8),
        ("int4", 4),
        ("int2", 2),
    ]

    print(f"{'precision':>10} | {'bytes/elem':>10} | {'MSE':>12} | {'SNR (dB)':>10}")
    print("-" * 50)

    rows = []
    for label, bits in configs:
        dtype_key = {"bf16/fp16": "fp16", "fp8": "fp8", "int8": "int8", "int4": "int4", "int2": "int2"}[label]
        reconstructed = quantize_dequantize(original, bits)
        m = mse(original, reconstructed)
        snr = snr_db(original, reconstructed)
        rows.append({
            "precision": label, "bits": bits,
            "bytes_per_element": DTYPE_BYTES[dtype_key],
            "mse": m, "snr_db": snr,
        })
        print(f"{label:>10} | {DTYPE_BYTES[dtype_key]:>10} | {m:>12.4f} | "
              f"{'inf' if snr == float('inf') else f'{snr:>10.1f}'}")

    print("\n[Article insight]")
    print("  SNR degrades sharply below int8, and the outlier channels are exactly why:")
    print("  naive min-max quantization has to stretch its range to cover a handful of")
    print("  large outlier values, which wastes most of its quantization levels on a")
    print("  narrow band most values never reach. This is precisely the motivation for")
    print("  KIVI's asymmetric per-channel (keys) / per-token (values) scaling and")
    print("  KVQuant's dense-and-sparse decomposition: isolate outliers so the rest of")
    print("  the distribution can be quantized tightly. A naive uniform scheme like the")
    print("  one simulated here is a reasonable stand-in for 'quantization without any")
    print("  outlier handling' -- real production methods do meaningfully better at the")
    print("  same bit width by handling outliers separately.")
    print("  NOTE: MSE/SNR here are proxies for reconstruction fidelity, not measured")
    print("  downstream perplexity or task accuracy -- that requires a real model.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E09_kv_quantization.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  Results saved -> results/E09_kv_quantization.json")


if __name__ == "__main__":
    main()
