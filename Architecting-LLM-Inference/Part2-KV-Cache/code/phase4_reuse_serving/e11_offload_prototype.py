"""
E11 -- GPU-Only vs. CPU-Offloaded Cache (manual prototype)
==============================================================
HONESTY NOTE (carried over from the article outline): this manual
`.to("cpu")` / `.to("cuda")` approach is a controlled PROTOTYPE, not a
production offload engine. Real offload engines (e.g. vLLM's CPU swap
space) use pinned memory, double buffering, and async transfer overlapped
with compute -- this script does none of that, so its numbers will look
worse than a real engine's. It's useful for building intuition about the
PCIe round-trip cost, not for citing as representative throughput numbers.

Requires: torch (works on CPU-only machines too, but the "offload" story
only means something if you actually have a GPU -- on CPU-only this will
just measure two no-op transfers of the same speed and isn't informative).

Run: python3 phase4_reuse_serving/e11_offload_prototype.py
"""

import time
import json
from pathlib import Path


def main():
    print("E11 -- GPU-Only vs. CPU-Offloaded Cache (manual prototype)\n")

    try:
        import torch
    except ImportError:
        print("  torch not installed in this environment -- this script is meant to run")
        print("  on your Mac or a RunPod GPU box with `pip install torch`.")
        print("  Skipping execution; see the docstring for what it would measure.")
        return

    if not torch.cuda.is_available():
        print("  No CUDA GPU detected. This experiment is only meaningful with a real")
        print("  GPU<->CPU transfer to measure (PCIe round trip). Skipping execution.")
        print("  Run this on a RunPod GPU instance to get real numbers.")
        return

    device = "cuda"
    # Simulate a KV cache block: (num_layers, 2, batch, heads, seq_chunk, head_dim)
    shape = (32, 2, 1, 8, 128, 128)   # ~8M elements, roughly one Llama-3-8B-GQA block at fp16
    tensor_gpu = torch.randn(shape, dtype=torch.float16, device=device)

    results = {}

    # -- GPU-resident baseline: no transfer, just touch the tensor --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = tensor_gpu.sum().item()
    torch.cuda.synchronize()
    results["gpu_resident_access_ms"] = (time.perf_counter() - t0) * 1000

    # -- Naive offload: move to CPU, then restore to GPU (no pinning, no overlap) --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tensor_cpu = tensor_gpu.to("cpu")
    torch.cuda.synchronize()
    results["offload_to_cpu_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tensor_restored = tensor_cpu.to(device)
    torch.cuda.synchronize()
    results["restore_to_gpu_ms"] = (time.perf_counter() - t0) * 1000

    results["total_offload_roundtrip_ms"] = results["offload_to_cpu_ms"] + results["restore_to_gpu_ms"]
    results["tensor_size_mb"] = tensor_gpu.element_size() * tensor_gpu.nelement() / (1024 ** 2)
    results["effective_bandwidth_gbps"] = (
        (results["tensor_size_mb"] / 1024) / (results["total_offload_roundtrip_ms"] / 1000)
        if results["total_offload_roundtrip_ms"] > 0 else None
    )

    print(f"  Tensor size:            {results['tensor_size_mb']:.1f} MB")
    print(f"  GPU-resident access:    {results['gpu_resident_access_ms']:.3f} ms")
    print(f"  Offload to CPU:         {results['offload_to_cpu_ms']:.3f} ms")
    print(f"  Restore to GPU:         {results['restore_to_gpu_ms']:.3f} ms")
    print(f"  Round trip total:       {results['total_offload_roundtrip_ms']:.3f} ms")
    if results["effective_bandwidth_gbps"]:
        print(f"  Effective bandwidth:    {results['effective_bandwidth_gbps']:.1f} GB/s "
              f"(unpinned, no overlap -- a real engine using pinned memory typically "
              f"gets much closer to the ~16 GB/s PCIe Gen4 x16 ceiling)")

    print("\n[Article insight]")
    print("  This naive round trip pays the full synchronous cost of both transfer")
    print("  directions with no overlap against compute -- exactly the gap a real")
    print("  offload engine closes with pinned memory and double buffering (issue the")
    print("  transfer for block N+1 while block N is still being used). Use this")
    print("  prototype to build intuition about WHERE the cost comes from, not as a")
    print("  substitute for a real engine's measured throughput penalty.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E11_offload_prototype.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved -> results/E11_offload_prototype.json")


if __name__ == "__main__":
    main()
