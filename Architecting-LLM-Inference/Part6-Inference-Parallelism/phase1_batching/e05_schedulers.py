"""
E05 — Scheduler Algorithm Comparison
======================================
Goal:   Compare FCFS vs Priority vs Shortest-Job-First scheduling.
        Show how scheduler choice affects P99 latency and starvation.
Hardware: MacBook (pure simulation)
"""

import time
import sys
import os
import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.workload_generator import WorkloadConfig, generate_synthetic_workload, Request
from benchmarks.metrics_collector import ExperimentMetrics, RequestMetrics


class SchedulerType(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    SJF = "sjf"               # Shortest Job First (by output length)
    SRTF = "srtf"             # Shortest Remaining Time First


@dataclass(order=True)
class QueueItem:
    key: float
    request: Request = field(compare=False)


class Scheduler:
    def __init__(self, scheduler_type: SchedulerType, max_batch: int = 8,
                 tpot_s: float = 0.015, prefill_s: float = 0.0005):
        self.scheduler_type = scheduler_type
        self.max_batch = max_batch
        self.tpot_s = tpot_s
        self.prefill_s = prefill_s

    def _priority_key(self, req: Request, current_time: float) -> float:
        if self.scheduler_type == SchedulerType.FCFS:
            return req.arrival_time                        # earliest first
        elif self.scheduler_type == SchedulerType.PRIORITY:
            return -req.priority + req.arrival_time * 0.001  # high priority first
        elif self.scheduler_type == SchedulerType.SJF:
            return req.expected_output_len                 # shortest output first
        elif self.scheduler_type == SchedulerType.SRTF:
            return req.expected_output_len                 # approximate with total len
        return req.arrival_time

    def run(self, requests: list[Request]) -> list[RequestMetrics]:
        heap: list[QueueItem] = []
        pending = sorted(requests, key=lambda r: r.arrival_time)
        idx = 0
        current_time = 0.0
        completed = []

        while idx < len(pending) or heap:
            # Add all arrived requests to heap
            while idx < len(pending) and pending[idx].arrival_time <= current_time:
                req = pending[idx]
                key = self._priority_key(req, current_time)
                heapq.heappush(heap, QueueItem(key=key, request=req))
                idx += 1

            if not heap:
                if idx < len(pending):
                    current_time = pending[idx].arrival_time
                continue

            # Select up to max_batch requests
            batch = []
            temp = []
            while heap and len(batch) < self.max_batch:
                item = heapq.heappop(heap)
                batch.append(item.request)
                temp.append(item)

            # Push back unselected
            for item in temp[self.max_batch:]:
                heapq.heappush(heap, item)

            # Simulate batch execution
            dispatch_time = current_time
            max_prompt = max(r.prompt_len for r in batch)
            prefill_time = max_prompt * self.prefill_s
            decode_time = max(r.expected_output_len for r in batch) * self.tpot_s

            end_time = dispatch_time + prefill_time + decode_time

            for req in batch:
                completed.append(RequestMetrics(
                    request_id=req.request_id,
                    prompt_len=req.prompt_len,
                    output_len=req.expected_output_len,
                    arrival_time=req.arrival_time,
                    start_time=dispatch_time,
                    first_token_time=dispatch_time + prefill_time,
                    end_time=dispatch_time + prefill_time +
                             req.expected_output_len * self.tpot_s,
                ))

            current_time = end_time

        return completed


def analyze_starvation(metrics: list[RequestMetrics],
                        requests: list[Request]) -> dict:
    """Detect long-waiting requests (starvation indicators)."""
    wait_times = [m.start_time - m.arrival_time for m in metrics]
    req_map = {r.request_id: r for r in requests}

    high_priority_waits = []
    low_priority_waits = []
    for m in metrics:
        req = req_map.get(m.request_id)
        wait = m.start_time - m.arrival_time
        if req and req.priority == 1:
            high_priority_waits.append(wait)
        else:
            low_priority_waits.append(wait)

    return {
        "max_wait_s": max(wait_times),
        "p99_wait_ms": sorted(wait_times)[int(len(wait_times) * 0.99)] * 1000,
        "starvation_count": sum(1 for w in wait_times if w > 5.0),
        "high_priority_wait_mean_ms": (
            statistics.mean(high_priority_waits) * 1000 if high_priority_waits else 0
        ),
        "low_priority_wait_mean_ms": (
            statistics.mean(low_priority_waits) * 1000 if low_priority_waits else 0
        ),
    }


def main():
    print("E05 — Scheduler Comparison: FCFS vs Priority vs SJF\n")

    cfg = WorkloadConfig(
        num_requests=200,
        arrival_rate=15.0,
        prompt_len_mean=256,
        prompt_len_std=200,   # high variance to stress scheduler
        output_len_mean=80,
        output_len_std=60,
    )
    requests = generate_synthetic_workload(cfg)

    schedulers = [
        SchedulerType.FCFS,
        SchedulerType.PRIORITY,
        SchedulerType.SJF,
        SchedulerType.SRTF,
    ]

    results = []
    for stype in schedulers:
        print(f"  Scheduler: {stype.value}...")
        sched = Scheduler(scheduler_type=stype)
        metrics = sched.run(requests)

        em = ExperimentMetrics(
            experiment_id=f"E05_{stype.value}",
            description=f"Scheduler: {stype.value}",
            config={"scheduler": stype.value},
        )
        em.start_wall_time = time.time()
        for m in metrics:
            em.add(m)
        em.finish()

        starvation = analyze_starvation(metrics, requests)
        summary = em.summary()
        summary.update(starvation)
        results.append(summary)

        print(f"    TTFT p99:       {summary.get('ttft_ms_p99', 0):.0f}ms")
        print(f"    Starvation:     {starvation['starvation_count']} requests waited >5s")
        print(f"    Hi-pri wait:    {starvation['high_priority_wait_mean_ms']:.0f}ms  "
              f"Lo-pri wait: {starvation['low_priority_wait_mean_ms']:.0f}ms")
        em.save("results")

    # Comparison table
    print("\n── Scheduler Comparison ───────────────────────────────────────────")
    print(f"{'Scheduler':>10} {'TTFT_p50':>10} {'TTFT_p99':>10} "
          f"{'HiPri_wait':>12} {'Starvation':>12}")
    for r in results:
        print(f"{r['config']['scheduler']:>10} "
              f"{r.get('ttft_ms_median', 0):>10.0f} "
              f"{r.get('ttft_ms_p99', 0):>10.0f} "
              f"{r.get('high_priority_wait_mean_ms', 0):>12.0f} "
              f"{r.get('starvation_count', 0):>12}")

    print("\n[Article insight]")
    print("  FCFS is simple and fair but ignores priority — bad for latency-SLO workloads.")
    print("  Priority scheduling reduces wait for high-pri requests but can starve low-pri.")
    print("  SJF minimizes average wait time but requires output length prediction.")
    print("  vLLM uses FCFS by default; production systems add priority tiers.")


if __name__ == "__main__":
    main()
