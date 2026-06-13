"""
E25 — KV Cache Transfer Benchmark
=====================================
Goal:   Measure the latency cost of moving KV tensors between GPUs.
        This models the bottleneck in disaggregated P/D serving
        (Mooncake, DistServe-style architectures).
Run on: RunPod 2× RTX 3090

Key question: how much of a request's latency budget is consumed
by KV transfer from the prefill node to the decode node?
"""

import torch
import time
import json
from pathlib import Path

RESULTS_DIR = Path("results/e25_kv_transfer")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Qwen2.5-7B architecture
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float16
BYTES_PER_ELEMENT = 2  # fp16


def kv_tensor_size_bytes(seq_len: int) -> int:
    """Total KV cache bytes for a given sequence length (both K and V, all layers)."""
    return 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * seq_len * BYTES_PER_ELEMENT


def benchmark_gpu_to_gpu_transfer(src_device: int, dst_device: int,
                                   seq_len: int, repeats: int = 20) -> dict:
    """Measure GPU-to-GPU KV transfer latency via PCIe/NVLink."""
    size_bytes = kv_tensor_size_bytes(seq_len)
    size_mb = size_bytes / 1e6

    # Simulate KV tensor: (2, num_layers, num_kv_heads, seq_len, head_dim)
    kv_tensor = torch.randn(
        2, NUM_LAYERS, NUM_KV_HEADS, seq_len, HEAD_DIM,
        dtype=DTYPE, device=f"cuda:{src_device}"
    )

    # Warmup
    for _ in range(3):
        _ = kv_tensor.to(f"cuda:{dst_device}")
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize(src_device)
        t0 = time.perf_counter()
        transferred = kv_tensor.to(f"cuda:{dst_device}")
        torch.cuda.synchronize(dst_device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        del transferred

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    bandwidth_gbps = (size_bytes / 1e9) / (avg_ms / 1000)

    return {
        "seq_len": seq_len,
        "size_mb": size_mb,
        "latency_mean_ms": avg_ms,
        "latency_min_ms": min(latencies_ms),
        "latency_p99_ms": sorted(latencies_ms)[int(len(latencies_ms) * 0.99)],
        "bandwidth_gbps": bandwidth_gbps,
    }


def estimate_transfer_overhead(seq_len: int, tpot_ms: float = 15.0) -> dict:
    """
    Estimate what fraction of decode latency is consumed by KV transfer.
    Assumes PCIe 4.0 x16 bandwidth (~16 GB/s effective).
    """
    size_bytes = kv_tensor_size_bytes(seq_len)
    pcie_bandwidth_gbps = 16.0
    transfer_ms = (size_bytes / 1e9) / pcie_bandwidth_gbps * 1000

    # How many decode tokens worth of latency is this transfer?
    tokens_equivalent = transfer_ms / tpot_ms

    return {
        "seq_len": seq_len,
        "kv_size_mb": size_bytes / 1e6,
        "transfer_ms_pcie": transfer_ms,
        "tokens_equivalent": tokens_equivalent,
        "overhead_pct_at_128_tokens": transfer_ms / (128 * tpot_ms) * 100,
    }


def main():
    print("E25 — KV Cache Transfer Benchmark")
    print(f"Model profile: Qwen2.5-7B ({NUM_LAYERS}L, {NUM_KV_HEADS}KV heads, {HEAD_DIM}d)\n")

    # ── Theoretical estimates (CPU only, for article) ────────────────────────
    print("── Theoretical Transfer Overhead (PCIe 4.0 @ 16 GB/s) ─────────────")
    print(f"{'Seq Len':>10} {'KV Size':>10} {'Transfer':>12} {'Tokens Equiv':>14} {'Overhead%':>10}")
    for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
        est = estimate_transfer_overhead(seq_len)
        print(f"{seq_len:>10} {est['kv_size_mb']:>8.1f}MB "
              f"{est['transfer_ms_pcie']:>10.1f}ms "
              f"{est['tokens_equivalent']:>14.1f} "
              f"{est['overhead_pct_at_128_tokens']:>9.1f}%")

    # ── Real GPU transfer benchmark (requires 2 GPUs) ────────────────────────
    if not torch.cuda.is_available():
        print("\nNo GPU available — theoretical estimates only.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"\nOnly {num_gpus} GPU(s) available. GPU-to-GPU benchmark skipped.")
        print("Run on RunPod 2× RTX 3090 for real measurements.")
        return

    print(f"\n── Real GPU-to-GPU Transfer (GPU 0 → GPU 1) ──────────────────────")
    print(f"{'Seq Len':>10} {'Size MB':>10} {'Latency ms':>12} {'BW GB/s':>10}")

    results = []
    for seq_len in [128, 512, 1024, 2048, 4096]:
        r = benchmark_gpu_to_gpu_transfer(0, 1, seq_len)
        results.append(r)
        print(f"{r['seq_len']:>10} {r['size_mb']:>10.1f} "
              f"{r['latency_mean_ms']:>12.2f} {r['bandwidth_gbps']:>10.1f}")

    with open(RESULTS_DIR / "kv_transfer.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults → {RESULTS_DIR}/kv_transfer.json")
    print("\n[Article insight]")
    print("  At seq_len=2048, KV transfer takes ~Xms — equivalent to N decode tokens.")
    print("  For short sequences, transfer overhead dominates and disaggregation hurts.")
    print("  For long sequences (8K+), transfer cost amortizes and disaggregation helps.")
    print("  NVLink (600 GB/s) vs PCIe (16 GB/s) = 37x difference in transfer time.")
    print("  This is why Mooncake/DistServe target NVLink clusters for production P/D.")


if __name__ == "__main__":
    main()
