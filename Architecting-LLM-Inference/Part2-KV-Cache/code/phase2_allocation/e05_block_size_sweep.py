"""
E05 -- Block-Size Sweep
==========================
Goal: How does the paged allocator's block_size choice trade off internal
      fragmentation against metadata/allocation overhead?

Phase A (this file, no GPU needed): pure simulation sweep over
         block_size in {8, 16, 32, 64, 128}, using the same PagedPool and
         workload as E04, measuring occupancy, internal fragmentation,
         and block-table size (a proxy for metadata overhead).
Phase B (`e05_block_size_sweep_runpod.sh`, requires a RunPod GPU + vLLM):
         real block-size sweep against a live vLLM server, measuring
         throughput, TPOT, and P99 latency -- not just fragmentation.
         NOTE: supported block sizes are runtime/kernel-version dependent.
         Recent vLLM releases only support block_size >= 16 on most
         backends -- don't assume 8 works without checking your version.

Run: python3 phase2_allocation/e05_block_size_sweep.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload
from phase2_allocation.e04_allocator_simulation import PagedPool, build_events

import heapq


def run_paged_sim_for_block_size(requests, capacity_tokens: int, block_size: int) -> dict:
    total_blocks = capacity_tokens // block_size
    pool = PagedPool(total_blocks=total_blocks, block_size=block_size)
    events = build_events(requests)
    heap = []
    internal_frag_samples = []
    used_blocks_samples = []
    block_table_entries_samples = []   # sum of blocks-per-sequence across active seqs (metadata proxy)

    for arrival, req_id, prompt_len, output_len, duration in events:
        while heap and heap[0][0] <= arrival:
            _, done_id = heapq.heappop(heap)
            pool.free(done_id)

        actual_tokens = prompt_len + output_len
        ok = pool.allocate(req_id, actual_tokens)
        if ok:
            heapq.heappush(heap, (arrival + duration, req_id))

        internal_frag_samples.append(pool.internal_fragmentation_tokens())
        used_blocks_samples.append(pool.used_blocks)
        block_table_entries_samples.append(sum(b for b, _ in pool.allocations.values()))

    avg_internal_frag = sum(internal_frag_samples) / len(internal_frag_samples)
    avg_used_blocks = sum(used_blocks_samples) / len(used_blocks_samples)
    avg_block_table_entries = sum(block_table_entries_samples) / len(block_table_entries_samples)

    return {
        "block_size": block_size,
        "total_blocks": total_blocks,
        "rejections": pool.rejections,
        "rejection_rate_pct": pool.rejections / len(requests) * 100,
        "avg_internal_frag_tokens": avg_internal_frag,
        "avg_occupancy_pct": avg_used_blocks / total_blocks * 100,
        "avg_block_table_entries": avg_block_table_entries,
        "max_internal_frag_per_seq": block_size - 1,
    }


def main():
    print("E05 -- Block-Size Sweep (simulation)\n")

    cfg = WorkloadConfig(
        num_requests=300, arrival_rate=8.0,
        prompt_len_mean=256, prompt_len_std=128,
        output_len_mean=150, output_len_std=100,
        seed=42,
    )
    requests = generate_synthetic_workload(cfg)
    capacity_tokens = 20_000

    print(f"Workload: {len(requests)} requests, capacity={capacity_tokens} tokens\n")
    print(f"{'block_size':>10} | {'rejections':>10} | {'occupancy%':>10} | "
          f"{'internal_frag':>14} | {'block_table_entries':>20}")
    print("-" * 78)

    results = []
    for block_size in (8, 16, 32, 64, 128):
        r = run_paged_sim_for_block_size(requests, capacity_tokens, block_size)
        results.append(r)
        print(f"{block_size:>10} | {r['rejections']:>10} | {r['avg_occupancy_pct']:>9.1f}% | "
              f"{r['avg_internal_frag_tokens']:>14.1f} | {r['avg_block_table_entries']:>20.1f}")

    print("\n[Article insight]")
    print("  Smaller blocks -> less internal fragmentation per sequence (bounded by")
    print("  block_size - 1 tokens), but more block-table entries to track per sequence")
    print("  (more metadata, more pointer chasing in the attention kernel). Larger")
    print("  blocks -> less metadata overhead, but more wasted tokens in each partially")
    print("  filled tail block. vLLM's default of 16 sits near the sweet spot for")
    print("  typical workloads; workloads with very short sequences benefit more from")
    print("  smaller blocks, and very long, low-churn sequences can tolerate larger ones.")
    print("  CAUTION: this simulation isolates the allocator's own tradeoff. Real GPU")
    print("  kernel locality effects (Phase B) can shift the sweet spot -- always")
    print("  verify block sizes are actually supported by your vLLM/kernel version.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E05_block_size_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved -> results/E05_block_size_sweep.json")


if __name__ == "__main__":
    main()
