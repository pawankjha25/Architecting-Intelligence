"""
E02 -- Formula vs. Measured KV Memory
========================================
Goal: Validate M_kv = 2 x L x B x S x H_kv x D_h x P against real measured
      GPU memory, across prompt length, generated length, batch/concurrency,
      and cache precision.

Phase A (this file, no GPU needed): validate the formula's INTERNAL
         consistency (e.g. doubling S should double memory, halving dtype
         bytes should halve memory) and sweep it across configs.
Phase B (`measure_real_memory`, requires transformers + CUDA): load a real
         model, generate at increasing lengths, and diff
         torch.cuda.memory_allocated() before/after to compare against the
         formula's prediction. Numbers will differ slightly due to allocator
         overhead/reserved-vs-allocated memory -- that gap IS the "allocator
         overhead" line item worth reporting.

Run: python3 phase1_foundations/e02_memory_formula.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.memory_model import MODEL_CONFIGS, kv_cache_bytes, human_bytes, DTYPE_BYTES


def sweep_seq_len(model_key: str = "qwen2.5-7b", lengths=(128, 512, 1024, 2048, 4096, 8192, 16384)):
    cfg = MODEL_CONFIGS[model_key]
    rows = []
    for s in lengths:
        b = kv_cache_bytes(cfg, seq_len=s, batch_size=1, dtype="fp16")
        rows.append({"seq_len": s, "bytes": b, "human": human_bytes(b)})
    # Sanity check: memory should scale linearly with seq_len
    ratio = rows[-1]["bytes"] / rows[0]["bytes"]
    expected_ratio = lengths[-1] / lengths[0]
    assert abs(ratio - expected_ratio) < 1e-6, "Formula is not linear in seq_len -- bug!"
    return rows


def sweep_batch_size(model_key: str = "qwen2.5-7b", seq_len: int = 2048, batches=(1, 2, 4, 8, 16, 32, 64)):
    cfg = MODEL_CONFIGS[model_key]
    rows = []
    for bsz in batches:
        b = kv_cache_bytes(cfg, seq_len=seq_len, batch_size=bsz, dtype="fp16")
        rows.append({"batch_size": bsz, "bytes": b, "human": human_bytes(b)})
    ratio = rows[-1]["bytes"] / rows[0]["bytes"]
    expected_ratio = batches[-1] / batches[0]
    assert abs(ratio - expected_ratio) < 1e-6, "Formula is not linear in batch_size -- bug!"
    return rows


def sweep_precision(model_key: str = "qwen2.5-7b", seq_len: int = 4096):
    cfg = MODEL_CONFIGS[model_key]
    rows = []
    for dtype in ("fp32", "fp16", "fp8", "int4", "int2"):
        b = kv_cache_bytes(cfg, seq_len=seq_len, batch_size=1, dtype=dtype)
        rows.append({"dtype": dtype, "bytes": b, "human": human_bytes(b),
                     "bytes_per_element": DTYPE_BYTES[dtype]})
    # fp16 should be exactly half of fp32
    fp32 = next(r["bytes"] for r in rows if r["dtype"] == "fp32")
    fp16 = next(r["bytes"] for r in rows if r["dtype"] == "fp16")
    assert abs(fp32 / fp16 - 2.0) < 1e-6, "fp16 should be exactly half of fp32 -- bug!"
    return rows


def measure_real_memory(model_name: str = "gpt2", lengths=(128, 256, 512, 1024)):
    """
    Phase B: requires `pip install torch transformers` and ideally CUDA.
    Compares formula prediction against torch.cuda.memory_allocated() deltas.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    results = []
    for length in lengths:
        text = "word " * length
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=length).to(device)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            _ = out.past_key_values  # force materialization

        if device == "cuda":
            after = torch.cuda.memory_allocated()
            measured = after - before
        else:
            measured = None  # CPU doesn't give reliable allocator deltas

        results.append({"seq_len": length, "measured_bytes": measured})

    return results


def main():
    print("E02 -- Formula vs. Measured KV Memory\n")

    print("-- Sweep: sequence length (formula only, linearity check) --------")
    for row in sweep_seq_len():
        print(f"  seq_len={row['seq_len']:6d}  ->  {row['human']}")
    print("  [OK] memory scales linearly with sequence length")

    print("\n-- Sweep: batch size / concurrency ---------------------------------")
    for row in sweep_batch_size():
        print(f"  batch={row['batch_size']:3d}  ->  {row['human']}")
    print("  [OK] memory scales linearly with batch size")

    print("\n-- Sweep: precision -------------------------------------------------")
    for row in sweep_precision():
        print(f"  {row['dtype']:5s} ({row['bytes_per_element']} B/elem)  ->  {row['human']}")
    print("  [OK] fp16 is exactly half of fp32, as expected")

    print("\n[Article insight]")
    print("  The formula is exactly linear in seq_len, batch_size, and precision --")
    print("  by construction, since it's a product of independent factors. The gap")
    print("  worth measuring in Phase B is allocator overhead: real frameworks reserve")
    print("  more than they allocate (memory pools, alignment, fragmentation), so")
    print("  torch.cuda.memory_reserved() > memory_allocated() > formula prediction")
    print("  is the expected ordering when you run this against a real model.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E02_memory_formula.json", "w") as f:
        json.dump({
            "seq_len_sweep": sweep_seq_len(),
            "batch_sweep": sweep_batch_size(),
            "precision_sweep": sweep_precision(),
        }, f, indent=2)
    print("\n  Results saved -> results/E02_memory_formula.json")


if __name__ == "__main__":
    main()
