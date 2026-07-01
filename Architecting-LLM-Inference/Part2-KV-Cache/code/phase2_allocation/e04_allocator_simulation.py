"""
E04 -- Contiguous vs. Paged Allocator Simulation
===================================================
Goal: Isolate ALLOCATOR behavior from GPU-kernel effects with a pure Python,
      event-driven simulation. No GPU, no model -- this experiment is about
      the memory-management data structures alone.

Two allocators, same total memory budget (in token-slots), same request
stream:

  ContiguousPool -- must reserve a CONTIGUOUS span for each sequence up
      front, sized to the worst case (max possible output length), because
      it cannot cheaply grow a contiguous allocation later. This produces:
        - internal fragmentation: reserved-but-never-used slots (most
          sequences finish long before hitting the worst-case length)
        - external fragmentation: holes between allocations of different
          sizes that no single new request fits into, even when aggregate
          free space would be enough

  PagedPool -- allocates fixed-size blocks on demand as tokens are actually
      generated (rounded up to block_size). This bounds internal
      fragmentation to at most (block_size - 1) tokens per sequence and
      eliminates external fragmentation by construction (blocks need not
      be contiguous).

Run: python3 phase2_allocation/e04_allocator_simulation.py
"""

import sys
import os
import json
import heapq
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload, Request

MAX_OUTPUT_LEN = 512   # worst-case output length a contiguous allocator must reserve for
BLOCK_SIZE = 16        # tokens per block (vLLM default)


# -- Contiguous allocator (first-fit interval allocator) -----------------------

@dataclass
class ContiguousPool:
    capacity: int
    free_intervals: list = field(default_factory=list)   # sorted list of [start, length)
    allocations: dict = field(default_factory=dict)      # req_id -> (start, length)
    rejections: int = 0

    def __post_init__(self):
        self.free_intervals = [(0, self.capacity)]

    def allocate(self, req_id: str, length: int) -> bool:
        """First-fit: find the first free interval big enough for `length`."""
        for i, (start, size) in enumerate(self.free_intervals):
            if size >= length:
                self.allocations[req_id] = (start, length)
                remaining = size - length
                if remaining > 0:
                    self.free_intervals[i] = (start + length, remaining)
                else:
                    self.free_intervals.pop(i)
                return True
        self.rejections += 1
        return False

    def free(self, req_id: str) -> None:
        if req_id not in self.allocations:
            return
        start, length = self.allocations.pop(req_id)
        self.free_intervals.append((start, length))
        self.free_intervals.sort()
        self._merge_adjacent()

    def _merge_adjacent(self) -> None:
        merged = []
        for start, size in self.free_intervals:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_start, prev_size = merged.pop()
                merged.append((prev_start, prev_size + size))
            else:
                merged.append((start, size))
        self.free_intervals = merged

    @property
    def used_capacity(self) -> int:
        return sum(length for _, length in self.allocations.values())

    @property
    def largest_free_block(self) -> int:
        return max((size for _, size in self.free_intervals), default=0)

    @property
    def total_free(self) -> int:
        return self.capacity - self.used_capacity

    def external_fragmentation_pct(self) -> float:
        """
        How much of the free space is unusable as one contiguous chunk.
        0% = all free space is in one block; high % = free space is scattered.
        """
        if self.total_free == 0:
            return 0.0
        return (1 - self.largest_free_block / self.total_free) * 100


# -- Paged allocator (block-based, ref-counting omitted -- see E06/E07 for CoW) -

@dataclass
class PagedPool:
    total_blocks: int
    block_size: int = BLOCK_SIZE
    free_block_ids: list = field(default_factory=list)
    allocations: dict = field(default_factory=dict)   # req_id -> (num_blocks, actual_tokens)
    rejections: int = 0

    def __post_init__(self):
        self.free_block_ids = list(range(self.total_blocks))

    def allocate(self, req_id: str, actual_tokens: int) -> bool:
        blocks_needed = -(-actual_tokens // self.block_size)  # ceil division
        if len(self.free_block_ids) < blocks_needed:
            self.rejections += 1
            return False
        for _ in range(blocks_needed):
            self.free_block_ids.pop()
        self.allocations[req_id] = (blocks_needed, actual_tokens)
        return True

    def free(self, req_id: str) -> None:
        if req_id not in self.allocations:
            return
        blocks_needed, _ = self.allocations.pop(req_id)
        for i in range(blocks_needed):
            self.free_block_ids.append(-1)  # any id works; we only track count here
        # keep free_block_ids length correct without needing real distinct ids
        self.free_block_ids = list(range(len(self.free_block_ids)))

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self.free_block_ids)

    def internal_fragmentation_tokens(self) -> int:
        return sum(
            blocks * self.block_size - actual
            for blocks, actual in self.allocations.values()
        )


# -- Event-driven simulation ---------------------------------------------------

TIME_PER_TOKEN = 0.02  # synthetic seconds/token for decode (arbitrary unit, consistent across both pools)


def build_events(requests: list[Request]):
    """Each request becomes an (arrival, req_id, prompt_len, output_len) tuple."""
    events = []
    for r in requests:
        duration = r.expected_output_len * TIME_PER_TOKEN
        events.append((r.arrival_time, r.request_id, r.prompt_len, r.expected_output_len, duration))
    return sorted(events)


def run_contiguous_sim(requests: list[Request], capacity_tokens: int) -> dict:
    pool = ContiguousPool(capacity=capacity_tokens)
    events = build_events(requests)
    heap = []  # (departure_time, req_id, reserved_length)
    frag_samples = []

    for arrival, req_id, prompt_len, output_len, duration in events:
        # process departures that happen before this arrival
        while heap and heap[0][0] <= arrival:
            _, done_id, _ = heapq.heappop(heap)
            pool.free(done_id)

        reserved = prompt_len + MAX_OUTPUT_LEN   # must reserve worst case, actual length unknown ahead of time
        reserved = min(reserved, capacity_tokens)
        ok = pool.allocate(req_id, reserved)
        if ok:
            heapq.heappush(heap, (arrival + duration, req_id, reserved))
        frag_samples.append(pool.external_fragmentation_pct())

    return {
        "allocator": "contiguous",
        "capacity_tokens": capacity_tokens,
        "rejections": pool.rejections,
        "total_requests": len(requests),
        "rejection_rate_pct": pool.rejections / len(requests) * 100,
        "avg_external_frag_pct": sum(frag_samples) / len(frag_samples) if frag_samples else 0,
    }


def run_paged_sim(requests: list[Request], capacity_tokens: int, block_size: int = BLOCK_SIZE) -> dict:
    total_blocks = capacity_tokens // block_size
    pool = PagedPool(total_blocks=total_blocks, block_size=block_size)
    events = build_events(requests)
    heap = []
    internal_frag_samples = []

    for arrival, req_id, prompt_len, output_len, duration in events:
        while heap and heap[0][0] <= arrival:
            _, done_id = heapq.heappop(heap)
            pool.free(done_id)

        actual_tokens = prompt_len + output_len   # paged only needs what's actually used
        ok = pool.allocate(req_id, actual_tokens)
        if ok:
            heapq.heappush(heap, (arrival + duration, req_id))
        internal_frag_samples.append(pool.internal_fragmentation_tokens())

    return {
        "allocator": "paged",
        "block_size": block_size,
        "capacity_tokens": capacity_tokens,
        "rejections": pool.rejections,
        "total_requests": len(requests),
        "rejection_rate_pct": pool.rejections / len(requests) * 100,
        "avg_internal_frag_tokens": sum(internal_frag_samples) / len(internal_frag_samples) if internal_frag_samples else 0,
        "external_fragmentation_pct": 0.0,   # by construction -- blocks need not be contiguous
    }


def main():
    print("E04 -- Contiguous vs. Paged Allocator Simulation\n")

    cfg = WorkloadConfig(
        num_requests=300, arrival_rate=8.0,
        prompt_len_mean=256, prompt_len_std=128,
        output_len_mean=150, output_len_std=100,
        seed=42,
    )
    requests = generate_synthetic_workload(cfg)

    # Capacity sized so paged can comfortably serve the workload but
    # contiguous's worst-case reservation makes it tight. At ~8 req/s
    # arrival and ~3s mean service time, average concurrency is ~24
    # requests; contiguous needs ~768 tokens/request reserved (prompt+512
    # worst case) vs. paged's ~406 tokens/request actually used.
    capacity_tokens = 20_000

    contiguous_result = run_contiguous_sim(requests, capacity_tokens)
    paged_result = run_paged_sim(requests, capacity_tokens)

    print(f"Workload: {len(requests)} requests, capacity={capacity_tokens} tokens\n")

    print("-- Contiguous allocator -------------------------------------------")
    print(f"  Rejections:              {contiguous_result['rejections']} / {contiguous_result['total_requests']}"
          f"  ({contiguous_result['rejection_rate_pct']:.1f}%)")
    print(f"  Avg external frag:        {contiguous_result['avg_external_frag_pct']:.1f}%")
    print(f"  (reserves prompt_len + {MAX_OUTPUT_LEN} worst-case tokens per request)")

    print("\n-- Paged allocator --------------------------------------------------")
    print(f"  Rejections:              {paged_result['rejections']} / {paged_result['total_requests']}"
          f"  ({paged_result['rejection_rate_pct']:.1f}%)")
    print(f"  Avg internal frag:        {paged_result['avg_internal_frag_tokens']:.1f} tokens "
          f"(summed across all concurrently active sequences; per-sequence bound is "
          f"block_size-1 = {BLOCK_SIZE - 1})")
    print(f"  External frag:            0% (by construction)")

    print("\n[Article insight]")
    print("  At the SAME token-capacity budget, the contiguous allocator rejects far")
    print("  more requests than the paged allocator -- not because there's less total")
    print("  memory, but because (a) it must reserve worst-case length per request")
    print("  since it can't grow a contiguous span cheaply, and (b) requests of")
    print("  different reserved sizes leave unusable holes (external fragmentation)")
    print("  even when aggregate free space would be enough. Paging removes both")
    print("  problems: allocate only what's actually used, and blocks never need to")
    print("  be contiguous, so there's no external fragmentation to speak of.")

    Path("results").mkdir(exist_ok=True)
    with open("results/E04_allocator_simulation.json", "w") as f:
        json.dump({"contiguous": contiguous_result, "paged": paged_result}, f, indent=2)
    print("\n  Results saved -> results/E04_allocator_simulation.json")


if __name__ == "__main__":
    main()
