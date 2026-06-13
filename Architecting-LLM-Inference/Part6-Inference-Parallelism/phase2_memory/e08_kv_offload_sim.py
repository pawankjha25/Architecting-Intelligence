"""
E08 — KV Cache Offloading Simulation
=======================================
Goal:   Model GPU→CPU→NVMe tiering latency tradeoffs.
        When does offloading help vs hurt throughput?
Hardware: MacBook (pure simulation — models real bandwidth)
"""

import sys
import os
import time
import random
import math
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload


# ── Hardware bandwidth constants (approximate real-world values) ───────────────

@dataclass
class HardwareProfile:
    name: str
    gpu_memory_gb: float
    gpu_bandwidth_gbps: float       # GPU HBM bandwidth
    gpu_cpu_bandwidth_gbps: float   # PCIe bandwidth (GPU ↔ CPU RAM)
    cpu_memory_gb: float
    cpu_nvme_bandwidth_gbps: float  # NVMe sequential bandwidth
    nvme_capacity_gb: float


# Representative hardware profiles
RTX_3090 = HardwareProfile(
    name="RTX 3090 + NVMe",
    gpu_memory_gb=24.0,
    gpu_bandwidth_gbps=936.0,
    gpu_cpu_bandwidth_gbps=16.0,    # PCIe 4.0 x16
    cpu_memory_gb=64.0,
    cpu_nvme_bandwidth_gbps=7.0,    # PCIe 4.0 NVMe
    nvme_capacity_gb=500.0,
)

A100_80G = HardwareProfile(
    name="A100 80GB + NVMe",
    gpu_memory_gb=80.0,
    gpu_bandwidth_gbps=2000.0,
    gpu_cpu_bandwidth_gbps=64.0,    # NVLink or PCIe 5.0
    cpu_memory_gb=512.0,
    cpu_nvme_bandwidth_gbps=14.0,
    nvme_capacity_gb=2000.0,
)


@dataclass
class ModelProfile:
    name: str
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: int = 2   # fp16

    @property
    def kv_size_per_token_bytes(self) -> int:
        """KV cache bytes per token (both K and V, all layers)."""
        return 2 * self.num_layers * self.num_heads * self.head_dim * self.dtype_bytes


LLAMA_8B = ModelProfile("Llama-3.1-8B", num_layers=32, num_heads=8, head_dim=128)
LLAMA_70B = ModelProfile("Llama-3.1-70B", num_layers=80, num_heads=8, head_dim=128)


class KVOffloadSimulator:
    """
    Simulates KV cache tiering across GPU → CPU → NVMe.
    Models:
      - How many tokens fit in GPU memory (after model weights)
      - Latency cost of fetching from CPU or NVMe on cache miss
      - Effective throughput with and without offloading
    """

    def __init__(self, hw: HardwareProfile, model: ModelProfile,
                 offload_cpu_gb: float = 0.0, offload_nvme_gb: float = 0.0):
        self.hw = hw
        self.model = model
        self.offload_cpu_gb = offload_cpu_gb
        self.offload_nvme_gb = offload_nvme_gb

        # Model weights take some GPU memory
        self.model_weights_gb = self._estimate_model_weights_gb()

        # Remaining GPU memory for KV cache
        kv_gpu_gb = hw.gpu_memory_gb - self.model_weights_gb
        self.gpu_kv_tokens = int(kv_gpu_gb * 1e9 / model.kv_size_per_token_bytes)
        self.cpu_kv_tokens = int(offload_cpu_gb * 1e9 / model.kv_size_per_token_bytes)
        self.nvme_kv_tokens = int(offload_nvme_gb * 1e9 / model.kv_size_per_token_bytes)
        self.total_kv_tokens = self.gpu_kv_tokens + self.cpu_kv_tokens + self.nvme_kv_tokens

    def _estimate_model_weights_gb(self) -> float:
        """Rough estimate: 2 bytes per param, ~12 params per (layer, head, head_dim)."""
        params = (self.model.num_layers * self.model.num_heads *
                  self.model.head_dim * 12)
        return (params * self.model.dtype_bytes) / 1e9

    def fetch_latency_s(self, token_count: int, from_tier: str) -> float:
        """Time to bring `token_count` KV tokens from a storage tier to GPU."""
        data_bytes = token_count * self.model.kv_size_per_token_bytes
        if from_tier == "gpu":
            return 0.0
        elif from_tier == "cpu":
            bw = self.hw.gpu_cpu_bandwidth_gbps * 1e9
            return data_bytes / bw
        elif from_tier == "nvme":
            bw = self.hw.cpu_nvme_bandwidth_gbps * 1e9
            return data_bytes / bw
        return 0.0

    def simulate_serving(self, requests, base_tpot_s: float = 0.015) -> dict:
        """
        Simulate serving requests with this memory configuration.
        Requests that exceed GPU KV capacity must offload/reload.
        """
        completed = 0
        total_latency = 0.0
        total_tokens = 0
        offload_penalties = []

        for req in requests:
            seq_len = req.prompt_len + req.expected_output_len

            # Does this sequence fit in GPU KV?
            if seq_len <= self.gpu_kv_tokens:
                tier = "gpu"
                fetch_lat = 0.0
            elif seq_len <= self.gpu_kv_tokens + self.cpu_kv_tokens:
                tier = "cpu"
                overflow = seq_len - self.gpu_kv_tokens
                fetch_lat = self.fetch_latency_s(overflow, "cpu")
            elif seq_len <= self.total_kv_tokens:
                tier = "nvme"
                overflow = seq_len - self.gpu_kv_tokens
                fetch_lat = self.fetch_latency_s(overflow, "nvme")
            else:
                # Exceeds all tiers — skip (OOM)
                continue

            decode_lat = req.expected_output_len * base_tpot_s
            total_lat = fetch_lat + decode_lat
            total_latency += total_lat
            total_tokens += req.expected_output_len
            offload_penalties.append(fetch_lat)
            completed += 1

        avg_offload_penalty_ms = (
            sum(offload_penalties) / len(offload_penalties) * 1000
            if offload_penalties else 0
        )
        throughput = total_tokens / total_latency if total_latency > 0 else 0

        return {
            "completed": completed,
            "total_kv_tokens": self.total_kv_tokens,
            "gpu_kv_tokens": self.gpu_kv_tokens,
            "throughput_tok_s": throughput,
            "avg_offload_penalty_ms": avg_offload_penalty_ms,
            "max_seq_len": self.total_kv_tokens,
        }


def main():
    print("E08 — KV Cache Offloading Simulation\n")

    cfg = WorkloadConfig(
        num_requests=200,
        prompt_len_mean=512,
        prompt_len_std=256,
        output_len_mean=256,
        output_len_std=128,
    )
    requests = generate_synthetic_workload(cfg)

    # ── Compare offload configurations ────────────────────────────────────────

    configs = [
        ("No offload",         0.0,  0.0),
        ("4GB CPU offload",    4.0,  0.0),
        ("8GB CPU offload",    8.0,  0.0),
        ("16GB CPU offload",  16.0,  0.0),
        ("8GB CPU + 8GB NVMe", 8.0,  8.0),
    ]

    print(f"Hardware: {RTX_3090.name} | Model: {LLAMA_8B.name}\n")
    print(f"{'Config':<25} {'Max Seq':>8} {'GPU KV Toks':>12} "
          f"{'Throughput':>12} {'Offload Penalty':>16}")
    print("-" * 80)

    for label, cpu_gb, nvme_gb in configs:
        sim = KVOffloadSimulator(RTX_3090, LLAMA_8B,
                                 offload_cpu_gb=cpu_gb, offload_nvme_gb=nvme_gb)
        result = sim.simulate_serving(requests)
        print(f"  {label:<23} {result['max_seq_len']:>8,} "
              f"{result['gpu_kv_tokens']:>12,} "
              f"{result['throughput_tok_s']:>10.0f}/s "
              f"{result['avg_offload_penalty_ms']:>13.1f}ms")

    print()
    print(f"Hardware: {RTX_3090.name} | Model: {LLAMA_70B.name}\n")
    for label, cpu_gb, nvme_gb in configs:
        sim = KVOffloadSimulator(RTX_3090, LLAMA_70B,
                                 offload_cpu_gb=cpu_gb, offload_nvme_gb=nvme_gb)
        result = sim.simulate_serving(requests)
        print(f"  {label:<23} {result['max_seq_len']:>8,} "
              f"completed={result['completed']:>4} "
              f"penalty={result['avg_offload_penalty_ms']:>6.1f}ms")

    print("\n[Article insight]")
    print("  CPU offloading extends context length but adds PCIe transfer latency.")
    print("  NVMe offloading extends further but ~10x slower than CPU transfer.")
    print("  For interactive serving: CPU offload OK, NVMe offload usually too slow.")
    print("  For batch/offline: NVMe offload viable when latency SLO is loose.")
    print("  The key insight: offloading trades throughput for capacity — pick based on SLO.")

    # ── Save results ──────────────────────────────────────────────────────────
    import json, os
    os.makedirs("results", exist_ok=True)
    offload_results = []
    for label, cpu_gb, nvme_gb in configs:
        for hw, model in [(RTX_3090, LLAMA_8B), (RTX_3090, LLAMA_70B)]:
            sim = KVOffloadSimulator(hw, model, offload_cpu_gb=cpu_gb, offload_nvme_gb=nvme_gb)
            r = sim.simulate_serving(requests)
            offload_results.append({
                "config": label,
                "hardware": hw.name,
                "model": model.name,
                "cpu_offload_gb": cpu_gb,
                "nvme_offload_gb": nvme_gb,
                "gpu_kv_tokens": r["gpu_kv_tokens"],
                "total_kv_tokens": r["total_kv_tokens"],
                "completed_requests": r["completed"],
                "throughput_tok_s": round(r["throughput_tok_s"], 1),
                "avg_offload_penalty_ms": round(r["avg_offload_penalty_ms"], 2),
            })

    save_data = {
        "experiment_id": "E08_kv_offload",
        "description": "KV cache offloading — GPU→CPU→NVMe latency tradeoffs",
        "config": {"num_requests": 200, "prompt_mean": 512, "output_mean": 256},
        "results": offload_results,
    }
    with open("results/E08_kv_offload.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("  Results saved → results/E08_kv_offload.json")


if __name__ == "__main__":
    main()
